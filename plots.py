# https://stackoverflow.com/questions/38161161/matplotlib-how-to-combine-multiple-bars-with-lines
# https://stackoverflow.com/questions/23293011/how-to-plot-a-superimposed-bar-chart-using-matplotlib-in-python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

mnb_dedep = (0.77214, 0.89647, 0.87438, 0.92312, 0.9279, 0.77214)
rf_dedep = (0.77183, 0.89744, 0.87391, 0.92467, 0.92794, 0.95255)
lsvc_dedep = (0.77273, 0.89525, 0.875, 0.92131, 0.92794, 0.31636)
ft_dedep = (0.77228, 0.89634, 0.87445, 0.92298, 0.92794, 0.47498)
lg_dedep = (0.77214, 0.89647, 0.87438, 0.92312, 0.92794, 0.6503)
db_dedep = (0.71005, 0.8558, 0.82717, 0.89125, 0.8971, 0.64317)

mnb_deindep = (0.77156, 0.89829, 0.87438, 0.92312, 0.9279, 0.77214)
rf_deindep = (0.77156, 0.89878, 0.89769, 0.89823, 0.89829, 0.85798)
lsvc_deindep = (0.87062, 0.86924, 0.8725, 0.87087, 0.87062, 0.82216)
ft_deindep = (0.92131, 0.92131, 0.92131, 0.92131, 0.92131, 0.88815)
lg_deindep = (0.9311, 0.93056, 0.93173, 0.93114, 0.9311, 0.90116)
db_deindep =  (0.77156, 0.94094, 0.2897, 0.443, 0.63576, 0.62774)

plt.figure(figsize=(10, 3.5))

ind = np.arange(6)  # number of metrics
rects1 = plt.bar(ind, mnb_dedep, 0.15, color='#0000e4', alpha=0.6)
rects2 = plt.bar(ind + 0.15, rf_dedep, 0.15, color='#0000ff',alpha=0.6)
rects3 = plt.bar(ind + 0.30, lsvc_dedep, 0.15, color='#0000e4',alpha=0.6)
rects4 = plt.bar(ind + 0.45, ft_dedep, 0.15, color='#0000ff',alpha=0.6)
rects5 = plt.bar(ind + 0.60, lg_dedep, 0.15, color='#0000e4',alpha=0.6)
rects6 = plt.bar(ind + 0.75, db_dedep, 0.15, color='#0000ff',alpha=0.6)

rects7 = plt.bar(ind, mnb_deindep, 0.15, color='#ffac16', alpha=0.6)
rects8 = plt.bar(ind + 0.15, rf_deindep, 0.15, color='#ffa500', alpha=0.6)
rects9 = plt.bar(ind + 0.30, lsvc_deindep, 0.15, color='#ffac16',alpha=0.6)
rects10 = plt.bar(ind + 0.45, ft_deindep, 0.15, color='#ffa500',alpha=0.6)
rects11 = plt.bar(ind + 0.60, lg_deindep, 0.15, color='#ffac16',alpha=0.6)
rects12 = plt.bar(ind + 0.75, db_deindep, 0.15, color='#ffa500', alpha=0.6)

'''
bargroups_dedep = {}
bargroups_deindep = {}
high_point_x = []
high_point_y = []

# 
for i in range(0,5):
    single_bargroup_dep={rects1[i].get_height():rects1[i].get_x() + rects1[i].get_width()/2.0,
                      rects2[i].get_height():rects2[i].get_x() + rects2[i].get_width()/2.0,
                      rects3[i].get_height():rects3[i].get_x() + rects3[i].get_width()/2.0,
                      rects4[i].get_height():rects4[i].get_x() + rects4[i].get_width()/2.0,
                      rects5[i].get_height():rects5[i].get_x() + rects4[i].get_width()/2.0,
                      rects6[i].get_height():rects6[i].get_x() + rects4[i].get_width()/2.0}
    bargroups_dedep.update(single_bargroup_dep)
    single_bargroup_indep={rects7[i].get_height():rects1[i].get_x() + rects1[i].get_width()/2.0,
                      rects8[i].get_height():rects2[i].get_x() + rects2[i].get_width()/2.0,
                      rects9[i].get_height():rects3[i].get_x() + rects3[i].get_width()/2.0,
                      rects10[i].get_height():rects4[i].get_x() + rects4[i].get_width()/2.0,
                      rects11[i].get_height():rects5[i].get_x() + rects4[i].get_width()/2.0,
                      rects12[i].get_height():rects6[i].get_x() + rects4[i].get_width()/2.0}
    bargroups_deindep.update(single_bargroup_indep)

y_dep = list(bargroups_dedep.keys())
x_dep = list(bargroups_dedep.values())
y_indep = list(bargroups_deindep.keys())
x_indep = list(bargroups_deindep.values())

#line_dep = plt.plot(x_dep,y_dep,marker='o', color='#0000ff', label='dependent')
#line_indep = plt.plot(x_indep,y_indep,marker='o', color='#ffa500', label='independent')

# plt.xlabel('Metrics')
# plt.ylabel('ms')
'''
x_positions=[]
p = 0
for i in range(len(ind)):
    x_positions.extend(ind+p)
    p = p + 0.15

x_positions = np.array(x_positions)
x_positions.sort()
plt.xticks(x_positions, ('MNB', 'RF', 'LSCV\naccuracy' ,'FT', 'LR', 'DB','MNB', 'RF', 'LSCV\nprecision', 'FT', 'LR', 'DB','MNB', 'RF', 'LSCV\nrecall','FT', 'LR', 'DB','MNB', 'RF', 'LSCV\nf1','FT', 'LR', 'DB','MNB', 'RF', 'LSCV\nauc','FT', 'LR', 'DB','MNB', 'RF', 'LSCV\nauprc','FT', 'LR', 'DB'), fontsize=6.5)
dependent_patch = mpatches.Patch(color='#0000ff', label='dependent')
independent_patch = mpatches.Patch(color='#ffa500', label='independent')
plt.legend(handles=[dependent_patch,independent_patch],prop={'size': 5.8}, loc='upper left')
plt.title("German")
plt.savefig('./plots/barplot1.png')
#plt.show()

