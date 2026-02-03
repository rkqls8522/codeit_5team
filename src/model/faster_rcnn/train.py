import torch
import os
from config import CONFIG
from model import get_model
from evaluate import evaluate


def train_one_epoch(model, data_loader, optimizer, device):
    """1 에포크 학습 수행. 개별 loss도 함께 반환."""
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
        optimizer.step()

        total_loss += losses.item()
        for k in loss_components:
            if k in loss_dict:
                loss_components[k] += loss_dict[k].item()

    n = len(data_loader)
    avg_loss = total_loss / n
    avg_components = {k: v / n for k, v in loss_components.items()}

    return avg_loss, avg_components


def train(model, train_loader, valid_loader, device, save_dir='checkpoints'):
    """전체 학습 루프."""
    os.makedirs(save_dir, exist_ok=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=CONFIG['learning_rate'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay']
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG['lr_scheduler_step'],
        gamma=CONFIG['lr_scheduler_gamma']
    )

    best_loss = float('inf')
    train_losses = []

    for epoch in range(CONFIG['num_epochs']):
        # 학습
        avg_loss, loss_components = train_one_epoch(
            model, train_loader, optimizer, device
        )
        lr_scheduler.step()
        train_losses.append(avg_loss)

        # 로그 출력
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch+1}/{CONFIG['num_epochs']}] "
              f"Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        print(f"  classifier: {loss_components['loss_classifier']:.4f} | "
              f"box_reg: {loss_components['loss_box_reg']:.4f} | "
              f"objectness: {loss_components['loss_objectness']:.4f} | "
              f"rpn_box_reg: {loss_components['loss_rpn_box_reg']:.4f}")

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

        # 매 epoch 체크포인트
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
