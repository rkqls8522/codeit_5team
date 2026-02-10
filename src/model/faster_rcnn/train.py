import torch
import os
from config import CONFIG
from evaluate import evaluate


def train_one_epoch(model, data_loader, optimizer, device, grad_clip_max_norm=None):
    """1 에포크 학습 수행. Gradient Clipping 지원."""
    model.train()
    total_loss = 0
    loss_components = {'loss_classifier': 0, 'loss_box_reg': 0,
                       'loss_objectness': 0, 'loss_rpn_box_reg': 0}

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()

        # Gradient Clipping (학습 안정성)
        if grad_clip_max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

        optimizer.step()

        total_loss += losses.item()
        for k in loss_components:
            if k in loss_dict:
                loss_components[k] += loss_dict[k].item()

    n = len(data_loader)
    avg_loss = total_loss / n
    avg_components = {k: v / n for k, v in loss_components.items()}

    return avg_loss, avg_components


def build_optimizer(model, config):
    """Optimizer 생성. Backbone 차등 LR 지원."""
    backbone_lr_ratio = config.get('backbone_lr_ratio', 1.0)
    lr = config['learning_rate']

    if backbone_lr_ratio < 1.0:
        # backbone과 head에 다른 LR 적용
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': lr * backbone_lr_ratio},
            {'params': head_params, 'lr': lr},
        ]
        print(f"  Backbone LR: {lr * backbone_lr_ratio:.6f} / Head LR: {lr:.6f}")
    else:
        param_groups = [p for p in model.parameters() if p.requires_grad]

    optimizer_type = config.get('optimizer', 'sgd')
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=config['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )

    print(f"  Optimizer: {optimizer_type.upper()}")
    return optimizer


def build_scheduler(optimizer, config):
    """LR 스케줄러 생성. Warmup + (Step|MultiStep|Cosine) 지원."""
    scheduler_type = config.get('lr_scheduler_type', 'step')
    warmup_epochs = config.get('warmup_epochs', 0)
    num_epochs = config['num_epochs']

    # 메인 스케줄러
    if scheduler_type == 'cosine':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-6
        )
    elif scheduler_type == 'multistep':
        # milestone에서 warmup 보정
        milestones = [m - warmup_epochs for m in config['lr_scheduler_milestones']
                      if m > warmup_epochs]
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=config['lr_scheduler_gamma']
        )
    else:  # step
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_scheduler_step'],
            gamma=config['lr_scheduler_gamma']
        )

    # Warmup 적용
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,  # LR * 0.01부터 시작
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
        print(f"  Scheduler: Warmup({warmup_epochs}ep) → {scheduler_type}")
    else:
        scheduler = main_scheduler
        print(f"  Scheduler: {scheduler_type}")

    return scheduler


def train(model, train_loader, valid_loader, device, save_dir='checkpoints'):
    """전체 학습 루프."""
    os.makedirs(save_dir, exist_ok=True)

    # Optimizer (차등 LR 지원)
    optimizer = build_optimizer(model, CONFIG)

    # LR Scheduler (Warmup 지원)
    lr_scheduler = build_scheduler(optimizer, CONFIG)

    # Gradient Clipping
    grad_clip = CONFIG.get('grad_clip_max_norm', None)
    if grad_clip:
        print(f"  Gradient Clipping: max_norm={grad_clip}")

    best_loss = float('inf')
    train_losses = []

    for epoch in range(CONFIG['num_epochs']):
        # 학습
        avg_loss, loss_components = train_one_epoch(
            model, train_loader, optimizer, device,
            grad_clip_max_norm=grad_clip
        )
        lr_scheduler.step()
        train_losses.append(avg_loss)

        # 로그 출력
        current_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[-1]['lr'] if len(optimizer.param_groups) > 1 else current_lr
        print(f"[Epoch {epoch+1}/{CONFIG['num_epochs']}] "
              f"Loss: {avg_loss:.4f} | LR: {current_lr:.6f} (head: {head_lr:.6f})")
        print(f"  cls: {loss_components['loss_classifier']:.4f} | "
              f"box: {loss_components['loss_box_reg']:.4f} | "
              f"obj: {loss_components['loss_objectness']:.4f} | "
              f"rpn: {loss_components['loss_rpn_box_reg']:.4f}")

        # 체크포인트 저장 (best loss 갱신 시)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  -> Best model saved (loss: {best_loss:.4f})")

        # 10 epoch마다 체크포인트 (앙상블용)
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_dir, f'epoch_{epoch+1}.pth'))

    # 학습 완료 후 loss 기록 저장
    with open(os.path.join(save_dir, 'train_losses.txt'), 'w') as f:
        for i, loss in enumerate(train_losses):
            f.write(f"Epoch {i+1}: {loss:.4f}\n")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    return train_losses
