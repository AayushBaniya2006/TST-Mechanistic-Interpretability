"""Generate a pipeline diagram showing the project workflow."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

def box(x, y, w, h, text, color, fontsize=10, bold=False):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='#333333', linewidth=1.5)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, wrap=True,
            color='#1a1a1a')

def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555',
                               lw=1.8, connectionstyle='arc3,rad=0'))

def arrow_curved(x1, y1, x2, y2, rad=0.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555',
                               lw=1.8, connectionstyle=f'arc3,rad={rad}'))

# Colors
blue = '#4A90D9'
light_blue = '#A8C8E8'
green = '#5CB85C'
light_green = '#A8DBA8'
orange = '#F0AD4E'
light_orange = '#F5D5A0'
red = '#D9534F'
light_red = '#E8A8A6'
purple = '#9B59B6'
light_purple = '#C9A0DC'
gray = '#EEEEEE'

# Title
ax.text(7, 9.6, 'Project Pipeline', ha='center', va='center',
        fontsize=16, fontweight='bold', color='#1a1a1a')

# Row 1: Input
box(0.5, 8.4, 3, 0.8, 'Time Series Input\n(JV / PenDigits / LSST)', light_blue, 9)
box(5.5, 8.4, 3, 0.8, 'TST Model\n(3 layers x 8 heads = 24)', light_blue, 9)
box(10.5, 8.4, 3, 0.8, 'Classification\n(9 / 10 / 96 classes)', light_blue, 9)
arrow(3.5, 8.8, 5.5, 8.8)
arrow(8.5, 8.8, 10.5, 8.8)

# Label: Original framework
ax.text(7, 7.9, 'Original Framework (Matiss)', ha='center', va='center',
        fontsize=8, fontstyle='italic', color='#888888')

# Row 2: Activation Patching
box(3.5, 7.0, 7, 0.8, 'Activation Patching: swap each head between correct & incorrect input,\nmeasure probability shift per head → 24 raw importance scores',
    light_orange, 9)
arrow(7, 8.4, 7, 7.8)

# Divider
ax.plot([0.3, 13.7], [6.7, 6.7], '--', color='#BBBBBB', lw=1)
ax.text(7, 6.45, 'My Extensions (Aayush)', ha='center', va='center',
        fontsize=8, fontstyle='italic', color='#888888')

# Row 3: Three validation branches
box(0.3, 5.0, 3.8, 1.2, 'Bootstrap CIs\n10,000 resamples per head\n→ 95% confidence intervals', light_green, 9)
box(5.1, 5.0, 3.8, 1.2, 'FDR Correction\nBenjamini-Hochberg\n→ adjusted p-values', light_green, 9)
box(9.9, 5.0, 3.8, 1.2, 'Stability Testing\nperturb input → re-patch\n→ rank correlation (rho)', light_green, 9)

arrow(5, 7.0, 2.2, 6.2)
arrow(7, 7.0, 7, 6.2)
arrow(9, 7.0, 11.8, 6.2)

# Row 4: Additional checks
box(0.3, 3.4, 3.8, 1.0, 'Effect Sizes\nCohen\'s d per head\n→ magnitude of effect', light_purple, 9)
box(5.1, 3.4, 3.8, 1.0, 'Baseline Comparisons\nIG, attention weights\n→ correlation with patching', light_purple, 9)
box(9.9, 3.4, 3.8, 1.0, 'Perturbation Types\nGaussian noise, time warp,\nphase shift', light_purple, 9)

arrow(2.2, 5.0, 2.2, 4.4)
arrow(7, 5.0, 7, 4.4)
arrow(11.8, 5.0, 11.8, 4.4)

# Row 5: Results
box(1.5, 1.5, 3.2, 1.2, 'JapaneseVowels\n21/24 significant\nd=3.35, rho=0.884', '#C8E6C9', 9, bold=True)
box(5.4, 1.5, 3.2, 1.2, 'PenDigits\n24/24 significant\nd=1.22, rho=0.894', '#C8E6C9', 9, bold=True)
box(9.3, 1.5, 3.2, 1.2, 'LSST\n0/24 significant\nd=0.50, rho=0.484', '#FFCDD2', 9, bold=True)

arrow(2.2, 3.4, 3.1, 2.7)
arrow(7, 3.4, 7, 2.7)
arrow(11.8, 3.4, 10.9, 2.7)

# Bottom label
ax.text(7, 0.9, 'Validated results with uncertainty quantification', ha='center', va='center',
        fontsize=10, fontstyle='italic', color='#555555')

plt.tight_layout()
plt.savefig('/Volumes/CS_Stuff/TST-Mechanistic-Interpretability/Results/Summary/figures/pipeline_diagram.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig('/Volumes/CS_Stuff/TST-Mechanistic-Interpretability/Results/Summary/figures/pipeline_diagram.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved pipeline diagram")
