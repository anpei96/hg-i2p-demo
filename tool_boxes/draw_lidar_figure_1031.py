'''
    ref blog
    https://zhuanlan.zhihu.com/p/701318388
'''

import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# 自定义数据
# ir data
'''
0.295	0.257	0.325	0.265	0.227	0.228
0.374	0.316	0.379	0.328	0.296	0.306
0.481	0.435	0.431	0.446	0.436	0.475
0.295	0.254	0.325	0.259	0.229	0.251
0.372	0.313	0.365	0.32	0.307	0.322
0.369	0.311	0.373	0.316	0.306	0.323
0.576	0.553	0.601	0.563	0.487	0.51
'''
df = pd.DataFrame({
'group': ['Baseline','FastMAC (cvpr24)','C','D','E','F','G'],
'sc-1': [0.295,0.374,0.481,0.295,0.372,0.369,0.576],
'sc-2': [0.257,0.316,0.435,0.254,0.313,0.311,0.553],
'sc-3': [0.325,0.379,0.431,0.325,0.365,0.373,0.601],
'sc-4': [0.265,0.328,0.446,0.259,0.320,0.316,0.563],
'sc-5': [0.227,0.296,0.436,0.229,0.307,0.306,0.487],
'sc-6': [0.228,0.306,0.475,0.251,0.322,0.323,0.510]
})

# rr data
'''
0.385	0.247	0.083	0.255	0.429	0.357
0.246	0.192	0.25	0.17	0.411	0.429
0.323	0.247	0.25	0.319	0.369	0.143
0.046	0.137	0.083	0.043	0.06	0.071
0.338	0.26	0.167	0.234	0.44	0.214
0.4	    0.329	0.167	0.298	0.435	0.429
0.415	0.493	0.5	    0.383	0.542	0.5
'''
# df = pd.DataFrame({
# 'group': ['Baseline','FastMAC (cvpr24)','C','D','E','F','G'],
# 'sc-1': [0.385,0.246,0.323,0.046,0.338,0.400,0.415],
# 'sc-2': [0.247,0.192,0.247,0.137,0.260,0.329,0.493],
# 'sc-3': [0.083,0.250,0.250,0.083,0.167,0.167,0.500],
# 'sc-4': [0.255,0.170,0.319,0.043,0.234,0.298,0.383],
# 'sc-5': [0.429,0.411,0.369,0.060,0.440,0.435,0.542],
# 'sc-6': [0.357,0.429,0.143,0.071,0.214,0.429,0.500]
# })

# 计算变量个数
categories=list(df)[1:]
N = len(categories)

# 仅绘制第一行数据的雷达图
values = df.loc[0].drop('group').values.flatten().tolist() # 获取第一行数据，剔除group
values += values[:1] # 闭合圆形图，需要在末尾增加一个与起始相同的值

# 计算每个轴的角度
angles = [n / float(N) * 2 * pi for n in range(N)] # 每个变量的角度位置
angles += angles[:1] # 闭合圆形图，需要在末尾增加一个与起始相同的值

# 初始化布局
ax = plt.subplot(111, polar=True)

# 偏移-将第一个轴位于顶部
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# 将每个变量绘制在极坐标上
plt.xticks(angles[:-1], categories)

# y标签
ax.set_rlabel_position(0)
# ir
plt.yticks([0.2,0.4,0.6], ["0.2","0.4","0.6"], color="grey", size=9)
plt.ylim(0,0.64)
# rr
plt.yticks([0.2,0.4,0.6], ["0.2","0.4","0.6"], color="grey", size=9)
plt.ylim(0,0.64)


# 添加多个极坐标图
# 绘制第一个图
values = df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Baseline (w/o pruning)")
ax.fill(angles, values, 'b', alpha=0.1)

# 绘制第二个图
values = df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="FastMAC (cvpr'24)")
ax.fill(angles, values, 'r', alpha=0.1)

values = df.loc[2].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="SC2-PCR (cvpr'22)")
ax.fill(angles, values, 'r', alpha=0.1)

values = df.loc[3].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="SC2-PCR++ (pami'23)")
ax.fill(angles, values, 'r', alpha=0.1)

values = df.loc[4].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="MAC (cvpr'23)")
ax.fill(angles, values, 'r', alpha=0.1)

values = df.loc[5].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Turboreg (iccv'25)")
ax.fill(angles, values, 'r', alpha=0.1)

values = df.loc[6].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Ours")
ax.fill(angles, values, 'r', alpha=0.1)

# 图例
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.show()
