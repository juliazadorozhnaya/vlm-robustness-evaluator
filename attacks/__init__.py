from .specific_string_attack import SpecificStringAttack
from .targeted_attack import TargetedAttack
from .untargeted_attack import UntargetedAttack
from .rec_targeted_exclusive import ExclusiveTargetedRecAttack

from optimizers.random_patch import RandomPatchImageProcessor

def get_attack(name, model, tokenizer, attack_cfg, optimizer_cfg):
    """
    Возвращает объект атаки по имени.
    """
    if name == "specific_string":
        processor = RandomPatchImageProcessor(attack_cfg["patch_size"])
        return SpecificStringAttack(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            attack_cfg=attack_cfg,
            optimizer_cfg=optimizer_cfg,
        )
    elif name == "targeted_attack":
        return TargetedAttack(model, tokenizer, optimizer_cfg["name"], optimizer_cfg)
    elif name == "untargeted_attack":
        return UntargetedAttack(model, tokenizer, optimizer_cfg["name"], optimizer_cfg)
    elif name == "rec_targeted_exclusive":
        return ExclusiveTargetedRecAttack(model, attack_cfg, optimizer_cfg)
    else:
        raise ValueError(f"Unknown attack: {name}")
