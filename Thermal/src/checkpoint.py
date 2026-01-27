"""
Checkpoint Management for LGT Training

Handles:
- Complete state saving (model, optimizer, scheduler, scaler, global_step)
- Safe loading with config validation
- Resume verification to prevent "Resume Kick" phenomenon
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    global_step: int
) -> Path:
    """
    Save complete training state to checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer with momentum state
        scheduler: LR scheduler with step state
        scaler: AMP GradScaler
        config: Training configuration (for validation on resume)
        metrics: Current training metrics
        global_step: Global step counter (critical for alpha warmup)
    
    Returns:
        Path to saved checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': config,
        'metrics': metrics,
        # 🔥 Critical: Save config hash for validation
        'config_hash': _compute_config_hash(config)
    }
    
    checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"✓ Saved checkpoint: {checkpoint_path} (step={global_step})")
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    current_config: Dict[str, Any],
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Load complete training state from checkpoint with validation.
    
    🔥 CRITICAL: This function prevents "Resume Kick" by:
    1. Restoring optimizer momentum state
    2. Restoring scheduler step position
    3. Validating config consistency
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to restore state
        scheduler: Scheduler to restore state
        scaler: GradScaler to restore state
        current_config: Current config for validation
        device: Device to load to
    
    Returns:
        Dict with 'start_epoch' and 'global_step'
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 1. Validate config consistency
    _validate_config(checkpoint.get('config', {}), current_config)
    
    # 2. Load model state (handle compiled model prefix)
    model_state_dict = checkpoint['model_state_dict']
    model_state_dict = _fix_state_dict_keys(model_state_dict, model)
    
    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    logger.info("✓ Model state loaded")
    
    # 3. 🔥 CRITICAL: Restore optimizer state (prevents momentum reset)
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✓ Optimizer state restored (momentum preserved)")
        except Exception as e:
            logger.error(f"⚠️ Failed to restore optimizer state: {e}")
            logger.error("  This may cause Resume Kick - consider restarting training")
    else:
        logger.warning("⚠️ No optimizer state in checkpoint - momentum will reset!")
    
    # 4. 🔥 CRITICAL: Restore scheduler state (prevents re-warmup)
    if 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("✓ Scheduler state restored (no re-warmup)")
        except Exception as e:
            logger.error(f"⚠️ Failed to restore scheduler state: {e}")
            # Fallback: manually step scheduler
            _manual_scheduler_advance(scheduler, checkpoint)
    else:
        logger.warning("⚠️ No scheduler state - attempting manual advance")
        _manual_scheduler_advance(scheduler, checkpoint)
    
    # 5. Restore scaler state
    if 'scaler_state_dict' in checkpoint:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info("✓ GradScaler state restored")
        except Exception as e:
            logger.warning(f"Failed to restore scaler state: {e}")
    
    # 6. Extract resume info
    start_epoch = checkpoint['epoch'] + 1
    global_step = checkpoint.get('global_step', 0)
    
    logger.info(f"✓ Fully resumed from epoch {checkpoint['epoch']}")
    logger.info(f"  Next epoch: {start_epoch}")
    logger.info(f"  Global step: {global_step}")
    
    return {
        'start_epoch': start_epoch,
        'global_step': global_step,
        'metrics': checkpoint.get('metrics', {})
    }


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the latest epoch checkpoint in directory."""
    epoch_checkpoints = sorted(checkpoint_dir.glob('epoch_*.pt'))
    if epoch_checkpoints:
        return epoch_checkpoints[-1]
    return None


def _compute_config_hash(config: Dict) -> str:
    """Compute hash of critical config values for validation."""
    critical_keys = [
        ('loss', 'w_style'),
        ('loss', 'structure_weight'),
        ('loss', 'edge_boost'),
        ('model', 'num_styles'),
        ('model', 'base_channels'),
    ]
    
    values = []
    for keys in critical_keys:
        val = config
        for k in keys:
            val = val.get(k, 'MISSING') if isinstance(val, dict) else 'MISSING'
        values.append(f"{'.'.join(keys)}={val}")
    
    return '|'.join(values)


def _validate_config(saved_config: Dict, current_config: Dict):
    """
    Validate critical config values match between saved and current.
    
    🔥 This prevents silent Loss scale changes that cause apparent spikes.
    """
    critical_checks = [
        ('loss.w_style', 
         saved_config.get('loss', {}).get('w_style'),
         current_config.get('loss', {}).get('w_style')),
        ('loss.structure_weight',
         saved_config.get('loss', {}).get('structure_weight'),
         current_config.get('loss', {}).get('structure_weight')),
        ('loss.edge_boost',
         saved_config.get('loss', {}).get('edge_boost'),
         current_config.get('loss', {}).get('edge_boost')),
    ]
    
    mismatches = []
    for name, saved, current in critical_checks:
        if saved is not None and current is not None and saved != current:
            mismatches.append(f"  {name}: saved={saved} vs current={current}")
    
    if mismatches:
        logger.warning("⚠️ CONFIG MISMATCH DETECTED - may cause Loss jump!")
        for m in mismatches:
            logger.warning(m)
        logger.warning("Consider using the saved config values.")


def _fix_state_dict_keys(state_dict: Dict, model: torch.nn.Module) -> Dict:
    """Fix state_dict keys for compiled/non-compiled model mismatch."""
    has_orig_mod_prefix = any(k.startswith('_orig_mod.') for k in state_dict.keys())
    model_is_compiled = hasattr(model, '_orig_mod')
    
    if has_orig_mod_prefix and not model_is_compiled:
        logger.info("Converting compiled checkpoint to non-compiled format")
        return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    elif not has_orig_mod_prefix and model_is_compiled:
        logger.info("Converting non-compiled checkpoint to compiled format")
        return {f'_orig_mod.{k}': v for k, v in state_dict.items()}
    
    return state_dict


def _manual_scheduler_advance(scheduler, checkpoint: Dict):
    """Manually advance scheduler if state_dict is missing."""
    global_step = checkpoint.get('global_step', 0)
    if global_step > 0:
        logger.info(f"Manually advancing scheduler by {global_step} steps...")
        for _ in range(global_step):
            try:
                scheduler.step()
            except Exception:
                break
        logger.info(f"✓ Scheduler advanced to step ~{global_step}")


def compare_configs(checkpoint_path: Path, config_path: Path) -> Dict[str, Any]:
    """
    Compare checkpoint config with local config file.
    
    Useful for debugging Resume Kick issues.
    
    Returns:
        Dict with 'matches', 'mismatches', 'missing' keys
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved_config = checkpoint.get('config', {})
    
    with open(config_path, 'r') as f:
        local_config = json.load(f)
    
    def flatten_dict(d, prefix=''):
        items = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, key))
            else:
                items.append((key, v))
        return items
    
    saved_flat = dict(flatten_dict(saved_config))
    local_flat = dict(flatten_dict(local_config))
    
    all_keys = set(saved_flat.keys()) | set(local_flat.keys())
    
    result = {'matches': [], 'mismatches': [], 'missing': []}
    
    for key in sorted(all_keys):
        saved_val = saved_flat.get(key, '<MISSING>')
        local_val = local_flat.get(key, '<MISSING>')
        
        if saved_val == '<MISSING>' or local_val == '<MISSING>':
            result['missing'].append(f"{key}: saved={saved_val}, local={local_val}")
        elif saved_val != local_val:
            result['mismatches'].append(f"{key}: saved={saved_val}, local={local_val}")
        else:
            result['matches'].append(key)
    
    return result
