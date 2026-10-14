"""
bott.py — 4×2 波特图：零极点对幅频/相频特性的影响

左右两张子图为一组（左：幅频特性，右：相频特性），自上而下依次为
单极点、单零点、共轭极点、共轭零点四种情形；每组子图共用一个标题，
幅频/相频通过 ylabel 区分。相位坐标以 45° 为单位画网格，
每个 45° 刻度（含 ±45°、±135°）都在坐标轴上标注。
参考：https://chat.deepseek.com/share/dxklfn3rft3zkiwf4g
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullLocator

# 设置中文字体和统一字号（对数轴刻度经 mathtext 渲染，
# 负号需 U+2212 字形，故用 Microsoft YaHei 优先，同目录脚本惯例）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

omega0 = 10.0                     # 极点/零点位置（转折频率、自然频率），rad/s
zeta = 0.3                        # 共轭对的阻尼比
omega = np.logspace(-1, 3, 2000)  # 0.1 ~ 1000 rad/s，覆盖转折频率两侧各两个十倍频程

def mag_db(H):
    """幅频特性：20lg|H(jω)|（dB）"""
    return 20 * np.log10(np.maximum(np.abs(H), 1e-12))

def phase_deg(H):
    """相频特性：解卷绕后的相位（°）"""
    return np.unwrap(np.angle(H)) * 180 / np.pi

# 四种情形：(标题, 传递函数, 幅频标记偏移, 幅频 y 范围, 相频 y 范围)
cases = [
    ('单极点对频响的影响',
     lambda s: 1 / (1 + s / omega0),
     +8.0, (-60, 10), (-100, 10)),
    ('单零点对频响的影响',
     lambda s: 1 + s / omega0,
     -10.0, (-10, 40), (-10, 100)),
    ('共轭极点对频响的影响',
     lambda s: omega0**2 / (s**2 + 2 * zeta * omega0 * s + omega0**2),
     +8.0, (-60, 15), (-200, 20)),
    ('共轭零点对频响的影响',
     lambda s: (s**2 + 2 * zeta * omega0 * s + omega0**2) / omega0**2,
     -10.0, (-15, 45), (-20, 200)),
]

fig, axes = plt.subplots(4, 2, figsize=(14, 16))

for i, (title, H, mag_off, mag_ylim, ph_ylim) in enumerate(cases):
    Hjw = H(1j * omega)
    ax_mag, ax_ph = axes[i, 0], axes[i, 1]

    # 一组子图共用一个标题（放在左侧幅频子图上）
    ax_mag.set_title(title)

    ax_mag.semilogx(omega, mag_db(Hjw), 'b-', linewidth=2)
    ax_ph.semilogx(omega, phase_deg(Hjw), 'r-', linewidth=2)

    # 标注 ω0（转折频率/自然频率）处的幅值：极点 -3 dB，零点 +3 dB，
    # 共轭极点 +4.4 dB，共轭零点 -4.4 dB（ζ = 0.3）
    m0 = mag_db(H(1j * omega0))
    ax_mag.plot(omega0, m0, 'ko', markersize=8)
    ax_mag.annotate(f'{m0:.1f} dB', xy=(omega0, m0),
                    xytext=(omega0 * 1.15, m0 + mag_off), ha='left', va='center')
    # 相位子图只标出 ω0 处的相位点，取值由 45° 网格读出（不写 45° 等标注）
    p0 = np.angle(H(1j * omega0)) * 180 / np.pi
    ax_ph.plot(omega0, p0, 'ko', markersize=8)

    for ax in (ax_mag, ax_ph):
        ax.axvline(omega0, color='k', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xlim(omega[0], omega[-1])
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    ax_mag.set_ylim(mag_ylim)
    ax_ph.set_ylim(ph_ylim)

    # 相位以 45° 为单位长度：45° 整倍处画刻度与网格线，
    # 所有刻度（含 ±45°、±135°）都在坐标轴上标注
    ticks = np.arange(np.ceil(ph_ylim[0] / 45) * 45,
                      np.floor(ph_ylim[1] / 45) * 45 + 1, 45)
    ax_ph.set_yticks(ticks)
    ax_ph.set_yticklabels([f'{int(v)}' for v in ticks])

    # x 对数轴取 2~9 次刻度，幅频子图 y 取 5 dB 次刻度；
    # 相频子图 y 方向不画次刻度（避免 22.5° 网格过密）
    for ax in (ax_mag, ax_ph):
        ax.xaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10)))
    ax_mag.yaxis.set_minor_locator(AutoMinorLocator())
    ax_ph.yaxis.set_minor_locator(NullLocator())

# 幅频/相频通过 ylabel 区分
for ax in axes[:, 0]:
    ax.set_ylabel('幅频特性 (dB)')
for ax in axes[:, 1]:
    ax.set_ylabel('相频特性 (°)')
for ax in axes[-1]:
    ax.set_xlabel(r'角频率 $\omega$ (rad/s)')

fig.tight_layout()

# 保存到 Figures 目录
script_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_dir, '../Figures')
os.makedirs(figures_dir, exist_ok=True)
fig.savefig(os.path.join(figures_dir, 'bott.jpeg'), dpi=200, bbox_inches='tight')
print(f"Figure saved to {os.path.join(figures_dir, 'bott.jpeg')}")

plt.show()
