# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:31:55 2020

@author: lackent
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_decision_graph(D1,D2=[]):
    
    rect_attributes=[(0.5,1),(0.75,0.5)]
    
    #check if a second matrix is provided or not
    if len(D2)==0:
        list_of_Ds=[D1]
    else:
        list_of_Ds=[D1,D2]
    
    #if the matrixes are big dont use edge for rectangles
    if len(D1)<=25:
        use_edge=True
    else:
        use_edge=False

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    for d in range(len(list_of_Ds)):
        D=list_of_Ds[d]
        coord_shift,width=rect_attributes[d]
        for a in range(len(D)):
            for b in range(len(D[a])):
                if a==len(D)-1:
                    edge_color='green' if use_edge else 'None'
                    ax_g=ax1.add_patch(
                            patches.Rectangle((a+coord_shift, b+coord_shift), width, width,
                                              facecolor='green',alpha=0.5,edgecolor=edge_color)) 
                else:
                    if D[a][b]:
                        edge_color='green' if use_edge else 'None'
                        ax_g=ax1.add_patch(
                            patches.Rectangle((a+coord_shift, b+coord_shift), width, width,
                                              facecolor='green',alpha=0.5,edgecolor=edge_color))
                    else:
                        edge_color='red' if use_edge else 'None'
                        ax_r=ax1.add_patch(
                            patches.Rectangle((a+coord_shift, b+coord_shift), width, width,
                                              facecolor='red',alpha=0.5,edgecolor=edge_color))
    plt.ylim((0.5,len(D)+0.5))
    plt.xlim((0.5,len(D)+0.5))
    plt.xlabel('Number of Observed Candidates')
    plt.ylabel('Relative Rank of the Last Candidate')
    plt.title('Optimal Strategy')
    plt.legend((ax_g,ax_r),['accept','reject'])
    return None


