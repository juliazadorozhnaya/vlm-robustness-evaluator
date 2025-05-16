import torch
from multiprocessing import Pool, cpu_count
from configs.config_loader import load_config
from hub.loader import load_model
from data.loaders import get_dataset_loader
from attacks import get_attack
from outputs.writer import save_predictions, save_metrics
from metrics.evaluator import MetricsEvaluator


def run_task_on_model(task_name, task_cfg, model_name, model_cfg, attack_cfgs, optimizer_cfgs, config):
    print(f"[INFO] Running task '{task_name}' with model '{model_name}'")

    # === Загрузка модели ===
    model = load_model(model_cfg)
    device = model.device

    # === Загрузка датасета ===
    dataset = get_dataset_loader(task_cfg, config["processing"])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["processing"]["batch_size"],
        num_workers=config["processing"]["num_workers"],
        pin_memory=config["processing"]["pin_memory"],
        shuffle=False
    )

    predictions_clean = []
    predictions_adv = []

    # === Инференс без атаки ===
    for batch in dataloader:
        for item in batch:
            image = item["image"].to(device).unsqueeze(0)
            prompt = item["text"]
            output = model.generate(image, prompt)
            predictions_clean.append({
                "text": output,
                "label": item["label"],
                "image": item["image"],
                "meta": item.get("meta", {}),
            })

    save_predictions(
        predictions_clean,
        model_name=model_name,
        task_name=task_name,
        data=predictions_clean,
        is_adv=False,
        output_cfg=config["output"]
    )

    # === Атаки и инференс после атаки ===
    for attack_name, attack_cfg in attack_cfgs.items():
        if attack_cfg["model"] != model_name:
            continue

        print(f"[INFO] Applying attack: {attack_name}")
        optimizer_cfg = optimizer_cfgs[attack_cfg["optimizer"]]
        attack = get_attack(attack_name, model, model.tokenizer, attack_cfg, optimizer_cfg)

        predictions_adv.clear()

        for batch in dataloader:
            images = torch.stack([item["image"] for item in batch]).to(device)
            prompts = [item["text"] for item in batch]
            adv_images = attack.run(images, prompts)

            for i in range(len(batch)):
                adv_pred = model.generate(adv_images[i:i+1], prompts[i])
                predictions_adv.append({
                    "text": adv_pred,
                    "label": batch[i]["label"],
                    "image": adv_images[i].cpu(),
                    "meta": batch[i].get("meta", {}),
                })

        save_predictions(
            predictions_adv,
            model_name=model_name,
            task_name=task_name,
            data=predictions_adv,
            is_adv=True,
            output_cfg=config["output"]
        )

        # === Подсчёт метрик ===
        evaluator = MetricsEvaluator(config["output"], config["output"]["output_dir"])
        result = evaluator.evaluate(
            task_name=task_name,
            task_cfg=task_cfg,
            predictions_clean=predictions_clean,
            predictions_adv=predictions_adv
        )
        save_metrics(result["adv"], task_name, attack_name, config)


def run_all():
    config = load_config("configs/config.yaml")
    tasks = config["tasks"]
    models = config["models"]
    attacks = config["attacks"]
    optimizers = config["optimizers"]

    jobs = []

    for task_name, task_cfg in tasks.items():
        for model_name, model_cfg in models.items():
            task_attacks = {
                name: cfg for name, cfg in attacks.items()
                if cfg["model"] == model_name and cfg.get("enabled", True)
            }
            if not task_attacks:
                continue

            jobs.append((task_name, task_cfg, model_name, model_cfg, task_attacks, optimizers, config))

    print(f"[INFO] Launching {len(jobs)} benchmark jobs in parallel (using {min(cpu_count(), len(jobs))} workers)...")

    with Pool(processes=min(cpu_count(), len(jobs))) as pool:
        pool.starmap(run_task_on_model, jobs)
