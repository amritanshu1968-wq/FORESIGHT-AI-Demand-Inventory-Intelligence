import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/risk/risk_register.csv')
reorders = df[df['risk_quadrant'] == 'Reorder Now'].sort_values('lost_margin_risk', ascending=False).head(5)
markdowns = df[df['risk_quadrant'] == 'Markdown / Clear'].sort_values('excess_capital_locked', ascending=False).head(5)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), dpi=300)
fig.patch.set_facecolor('#FFFFFF')

# Reorders Table
ax1.axis('off')
ax1.set_title('Top Priority Replenishment Actions (Imminent Stockout Risk)', fontsize=11, fontweight='bold', color='#B91C1C', pad=10, loc='left')
cols1 = ['SKU ID', 'Product Category', 'On Hand', 'ROP', 'ROQ (Units)', 'Lost Margin Exposure']
data1 = []
for _, r in reorders.iterrows():
    val = f"Rs. {r['lost_margin_risk']:,.2f}"
    data1.append([r['sku_id'], r['category'], str(int(r['current_on_hand'])), str(int(r['reorder_point'])), str(int(r['recommended_order_qty'])), val])

t1 = ax1.table(cellText=data1, colLabels=cols1, loc='center', cellLoc='center')
t1.auto_set_font_size(False)
t1.set_fontsize(8.5)
t1.scale(1, 1.4)
for (row, col), cell in t1.get_celld().items():
    if row == 0:
        cell.set_facecolor('#FEE2E2')
        cell.set_text_props(weight='bold', color='#991B1B')
    else:
        cell.set_facecolor('#FEF2F2' if row % 2 == 1 else '#FFFFFF')

# Markdowns Table
ax2.axis('off')
ax2.set_title('Top Priority Liquidation Actions (Excess Capital Locked)', fontsize=11, fontweight='bold', color='#7E22CE', pad=10, loc='left')
cols2 = ['SKU ID', 'Product Category', 'On Hand', 'WOC', 'Excess Units', 'Capital Locked']
data2 = []
for _, r in markdowns.iterrows():
    val = f"Rs. {r['excess_capital_locked']:,.2f}"
    woc_str = f"{r['weeks_of_cover']:.1f}w"
    data2.append([r['sku_id'], r['category'], str(int(r['current_on_hand'])), woc_str, str(int(r['excess_units'])), val])

t2 = ax2.table(cellText=data2, colLabels=cols2, loc='center', cellLoc='center')
t2.auto_set_font_size(False)
t2.set_fontsize(8.5)
t2.scale(1, 1.4)
for (row, col), cell in t2.get_celld().items():
    if row == 0:
        cell.set_facecolor('#F3E8FF')
        cell.set_text_props(weight='bold', color='#6B21A8')
    else:
        cell.set_facecolor('#FAF5FF' if row % 2 == 1 else '#FFFFFF')

plt.tight_layout()
plt.savefig('outputs/charts/priority_actions_table.png', dpi=300, bbox_inches='tight')
print('Saved priority actions table successfully.')
