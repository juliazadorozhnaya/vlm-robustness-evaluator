import multiprocessing
from attacks import get_attack
from outputs.writer import save_predictions
from metrics.evaluator import MetricsEvaluator
from utils.logger import setup_logger


def run_single_attack(
    attack_name, attack_cfg, optimizer_cfg,
    model, dataset, task_cfg, config,
    task_name, model_name, clean_preds
):
    logger = setup_logger(f"attack-{attack_name}")
    logger.info(f"[{task_name}] Starting attack: {attack_name}")

    # === Генерация состязательных примеров ===
    attack = get_attack(attack_name, model, attack_cfg, optimizer_cfg)
    adv_preds = attack.run_dataset(dataset, task_cfg)

    if not adv_preds:
        logger.warning(f"[{task_name}] No adversarial predictions generated for attack '{attack_name}'")
        return

    # === Сохранение атакованных предсказаний ===
    save_predictions(
        data=adv_preds,
        output_cfg=config["output"],
        model_name=model_name,
        task_name=task_name,
        is_adv=True
    )

    # === Подсчёт метрик на атакованных данных ===
    logger.info(f"[{task_name}] Evaluating adversarial metrics...")
    evaluator = MetricsEvaluator(
        output_cfg=config["output"],
        output_path=config["output"]["output_dir"]
    )
    evaluator.evaluate(
        task_name=task_name,
        task_cfg=task_cfg,
        predictions_clean=clean_preds,
        predictions_adv=adv_preds
    )

    logger.info(f"[{task_name}] Attack {attack_name} completed.")


def run_all_tasks(config, model_registry, datasets):
    logger = setup_logger("controller")

    tasks_cfg = config["tasks"]
    attacks_cfg = config["attacks"]
    optimizers_cfg = config["optimizers"]
    num_workers = config["processing"].get("num_workers", multiprocessing.cpu_count())

    jobs = []

    for task_name, task_cfg in tasks_cfg.items():
        dataset = datasets[task_name]
        model_name = task_cfg["model"]
        model = model_registry[model_name]

        logger.info(f"[{task_name}] Running clean inference...")
        clean_preds = model.predict(dataset, task_cfg)

        save_predictions(
            data=clean_preds,
            output_cfg=config["output"],
            model_name=model_name,
            task_name=task_name,
            is_adv=False
        )

        save_flat_adv_predictions(
            output_root=config["output"]["output_dir"],
            optimizer=attack_cfg["optimizer"],
            model_name=model_name,
            task_name=task_name,
            attack_name=attack_name,
            prediction_data=adv_preds
        )

        logger.info(f"[{task_name}] Evaluating clean metrics...")
        evaluator = MetricsEvaluator(
            output_cfg=config["output"],
            output_path=config["output"]["output_dir"]
        )
        evaluator.evaluate(
            task_name=task_name,
            task_cfg=task_cfg,
            predictions_clean=clean_preds,
            predictions_adv=clean_preds
        )

        for attack_name, attack_cfg in attacks_cfg.items():
            optimizer_cfg = optimizers_cfg[attack_cfg["optimizer"]]

            p = multiprocessing.Process(
                target=run_single_attack,
                args=(
                    attack_name, attack_cfg, optimizer_cfg,
                    model, dataset, task_cfg, config,
                    task_name, model_name, clean_preds
                )
            )
            jobs.append(p)

    for i in range(0, len(jobs), num_workers):
        batch = jobs[i:i + num_workers]
        for p in batch:
            p.start()
        for p in batch:
            p.join()

    logger.info("All tasks completed.")
