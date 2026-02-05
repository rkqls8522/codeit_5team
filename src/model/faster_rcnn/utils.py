import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_prediction(image, boxes, labels, scores=None, class_names=None):
    """이미지 위에 바운딩 박스와 라벨을 시각화한다."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        text = class_names[label] if class_names else str(label.item())
        if scores is not None:
            text += f' {scores[i]:.2f}'
        ax.text(x1, y1 - 5, text, color='red', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.5))

    ax.axis('off')
    plt.tight_layout()
    plt.show()
