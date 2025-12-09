import os
import pandas as pd
import numpy as np
# import getpass
from lgt_df_utilities.lgt_utilities import User_Info
from lgt_df_utilities.lgt_df import LGT_df_General
# from lgt_wafer_utilities import lgt_wafer_utilitis
from lgt_boxes_generator.lgt_box_lib import Small_Box, Big_Box,Gel_Pak
import plotly.graph_objects as go
# import plotly.io as pio

def is_light_color(hex_color: str) -> bool:
    # Retire le # si présent
    hex_color = hex_color.lstrip('#')
    
    # Convertit en valeurs RGB (0-255)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # Calcule la luminance perçue
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    if luminance >= 128 :
           return ("#000001","#0D1CEE")
    else: 
           return ("#F8F9FA","#6FD6FB")
#     # Détermine si c'est clair ou foncé
#     return luminance >= 128  # True = clair, False = foncé

class Big_Box_Die_Display():
    @staticmethod
    def diplay_die_in_big_box(box_data,box_id, lot_id, defect_table_df, save_fig=True,QR=True,without_wafer=False):
        delta_x=81.5
        text_size=12
        fig = go.Figure()
        display_case_df=pd.DataFrame([])
        fig=Big_Box.lgt_big_box_generator(fig, box_id=box_id,lgt_oa_id=lot_id,defect_observed=not defect_table_df.empty,visual_axis=False)
        if not without_wafer:
                wafer_list=box_data['W_id'].unique().tolist()

                Wafer_df=pd.DataFrame({'text':wafer_list,
                                        'x':delta_x*np.ones(len(wafer_list)),
                                        'y':[54.5-i*1.82 for i in range(len(wafer_list))]})

                for index, row in Wafer_df.iterrows():
                        fig.add_annotation(x=row['x'], y=row['y'],text=row['text'],font=dict(size=16,color='gray'),showarrow=False,xanchor="left", align="left")
                
        if not defect_table_df.empty:
                for index, row in defect_table_df.iterrows():
                        fig.add_annotation(x=row['xb'], y=row['yb'],text=row['defect'],font=dict(size=16,color='gray'),showarrow=False,xanchor="left", align="left")

        for index, row in box_data.iterrows():
                fig.add_trace(go.Scatter(x=row[['left','left','right','right','left']].values, 
                                y=row[['bottom','top','top','bottom','bottom']].values,
                                fill='toself',
                                fillcolor = row['lgt_color_die'],
                                hoveron='points',
                                line_color=row['lgt_color_die'],
                                marker=dict(size=0, color='gold', opacity=0),showlegend=False))
                # number of case                    
                if row['width']>15 and 6>row['height']:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[1]})
                        # print(f'case {1}')
                        fig.add_annotation(x=row['left']+0.15*row['width'], y=row['bottom']+0.85*row['height'],text=index,font=dict(size=text_size+4,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['x']-0.15*row['width'], y=row['y']+0.25*row['height'],text=row['WFC'][:2],textangle=0,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x']-0.15*row['width'], y=row['y']-0.25*row['height'],text=row['WFC'][2:],textangle=0,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x']+0.25*row['width'], y=row['y'],text=row["all_defect"],textangle=0,font=dict(size=text_size,color='gray'),showarrow=False)

                elif 15>=row['width']>=8 and 10>row['height']>=3:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[2]})
                        # print(f'case {2}')
                        fig.add_annotation(x=row['left']+0.1*row['width'], y=row['top']-0.25*row['height'],text=index,font=dict(size=text_size+2,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y']+0.05*row['height'],text=row['WFC'],textangle=0,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y']-0.25*row['height'],text=row["all_defect"],textangle=0,font=dict(size=text_size,color='gray'),showarrow=False)
                elif 15>=row['width']>=8 and 3>row['height']:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[3]})
                        # print(f'case {2}')
                        fig.add_annotation(x=row['left']+0.1*row['width'], y=row['top']-0.25*row['height'],text=index,font=dict(size=text_size+2,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y']+0.15*row['height'],text=row['WFC'],textangle=0,font=dict(size=text_size+1,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y']-0.35*row['height'],text=row["all_defect"],textangle=0,font=dict(size=text_size-1,color='gray'),showarrow=False)
                elif 8>row['width']>=4 and 7>row['height']:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[4]})
                        fig.add_annotation(x=row['left']+0.15*row['width'], y=row['bottom']+0.85*row['height'],text=index,font=dict(size=text_size,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['left']+0.7*row['width'], y=row['bottom']+0.8*row['height'],text=row['W'],textangle=0,font=dict(size=text_size,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y'],text="F"+str(row['F'])+row['D'],textangle=0,font=dict(size=text_size+1,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']+0.15*row['height'],text=row["all_defect"],textangle=0,font=dict(size=text_size,color='gray'),showarrow=False)
                
                elif 4>row['width']>=3 and 5>row['height']:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[5]})
                        fig.add_annotation(x=row['x']+0.15*row['width'], y=row['top']+15*row['height'],text=index,font=dict(size=text_size+1,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y']+0.1*row['height'],text=row['W'],textangle=0,font=dict(size=text_size,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['left']-0.8, y=row['y'],text="F"+str(row['F'])+row['D'],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        # fig.add_annotation(x=row['right']+1, y=row['y'],text="D"+row['WFC'][2:].split('D')[1],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']-0.8,text=row["all_defect"],textangle=0,font=dict(size=text_size-1,color='gray'),showarrow=False)

                elif 3>row['width']>=1 and 5>row['height']:
                        # print(f'case {8}')
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[6]})
                        fig.add_annotation(x=row['x'], y=row['top']-1.2,text=index,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['left']-0.6, y=row['y'],text=row['WFC'][:2],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['top']+0.6,text=row['WFC'][2:].split('D')[0],textangle=0,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['right']+1, y=row['y'],text="D"+row['WFC'][2:].split('D')[1],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']-0.8,text=row["all_defect"],textangle=0,font=dict(size=text_size-1,color='gray'),showarrow=False)
                elif 1>row['width'] and 10>row['height']:
                        # print(f'case {8}')
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[7]})
                        fig.add_annotation(x=row['x'], y=row['y'],text=index,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['right']+1, y=row['top']-0.15*row['height'],text=row['WFC'][:2].split('D')[0],textangle=0,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['left']-1, y=row['y'],text=row['WFC'][2:],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        # fig.add_annotation(x=row['x'], y=row['top']+0.6,text=row['WFC'][2:].split('D')[0],textangle=0,font=dict(size=text_size,color='white'),showarrow=False)
                        # fig.add_annotation(x=row['right']+1, y=row['y'],text="D"+row['WFC'][2:].split('D')[1],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['right']+1, y=row['y'],text=row["all_defect"],textangle=-90,font=dict(size=text_size-1,color='gray'),showarrow=False)
                else:
                        # print(f'case {8}')
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[8]})
                        fig.add_annotation(x=row['left']+0.15*row['width'], y=row['bottom']+0.85*row['height'],text=index,font=dict(size=text_size+4,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['left']+0.7*row['width'], y=row['bottom']+0.8*row['height'],text=row['W'],textangle=0,font=dict(size=text_size+6,color=row['lgt_gray_text']),showarrow=False)
                        # fig.add_annotation(x=row['x'], y=row['bottom']+0.65*row['height'],text="F"+str(row['F'])+row['D'],textangle=0,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        # fig.add_annotation(x=row['x'], y=row['bottom']+0.45*row['height'],text="F"+str(row['F'])+row['D'],textangle=0,font=dict(size=text_size+8,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']+0.45*row['height'],text=str(row['F'])+row['D'],textangle=0,font=dict(size=text_size+8,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']+0.15*row['height'],text=row["all_defect"],textangle=0,font=dict(size=text_size+4,color='gray'),showarrow=False)

        # pre-binning
        user_google_drive=User_Info.get_user_GoogleDrive()
        if save_fig:
                if QR=='Quality Report':
                        os.chdir(r''+user_google_drive+'/Engineering - Lots/'+lot_id+'/Quality reports')
                        fig.write_image(box_id+"-QR.png",format='png',width=1183,height=900,scale=3)
                else:
                        os.chdir(r''+ user_google_drive +'/Engineering - Lots/'+lot_id+'/Box maps')
                        fig.write_image(box_id+"-BM.png",format='png',width=1183,height=900,scale=3)

        return (fig,display_case_df)
        # return fig
class Gel_Pak_Die_Display():
    @staticmethod
    def diplay_die_in_big_box(box_data,box_id, lot_id, defect_table_df, save_fig=True,QR=True,without_wafer=True):
        delta_x=81.5
        text_size=12
        fig = go.Figure()
        display_case_df=pd.DataFrame([])
        fig=Gel_Pak.lgt_big_box_generator(fig, box_id=box_id,lgt_oa_id=lot_id,defect_observed=not defect_table_df.empty,visual_axis=False)
        if not without_wafer:
                wafer_list=box_data['W_id'].unique().tolist()

                Wafer_df=pd.DataFrame({'text':wafer_list,
                                        'x':delta_x*np.ones(len(wafer_list)),
                                        'y':[54.5-i*1.82 for i in range(len(wafer_list))]})

                for index, row in Wafer_df.iterrows():
                        fig.add_annotation(x=row['x'], y=row['y'],text=row['text'],font=dict(size=16,color='gray'),showarrow=False,xanchor="left", align="left")
                
        if not defect_table_df.empty:
                for index, row in defect_table_df.iterrows():
                        fig.add_annotation(x=row['xb'], y=row['yb'],text=row['defect'],font=dict(size=14,color='gray'),showarrow=False,xanchor="left", align="left")

        for index, row in box_data.iterrows():
                text_color,index_color=is_light_color(row["lgt_color_die"])

                # fig.add_trace(go.Scatter(x=row[['x','x']].values, 
                #                 y=[72.8,78],
                #                 mode='lines',
                #                 marker=dict(size=2, color="#7E8080", opacity=0),showlegend=False))
                # fig.add_trace(go.Scatter(x=row[['x','x']].values, 
                #                 y=[-6,-0.8],
                #                 mode='lines',
                #                 marker=dict(size=2, color="#7E8080", opacity=0),showlegend=False))
                # fig.add_trace(go.Scatter(x=[72.8,78], 
                #                 y=row[['y','y']].values,
                #                 mode='lines',
                #                 marker=dict(size=2, color="#7E8080", opacity=0),showlegend=False))
                # fig.add_trace(go.Scatter(x=[-6,-0.8], 
                #                 y=row[['y','y']].values,
                #                 mode='lines',
                #                 marker=dict(size=2, color="#7E8080", opacity=0),showlegend=False))
                
                fig.add_trace(go.Scatter(x=row[['left','left','right','right','left']].values, 
                                y=row[['bottom','top','top','bottom','bottom']].values,
                                fill='toself',
                                fillcolor = row['lgt_color_die'],
                                hoveron='points',
                                line_color=row['lgt_color_die'],
                                marker=dict(size=0, color='gold', opacity=0),showlegend=False))


                if row['width']>15 and 6>row['height']:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[1]})
                        # print(f'case {1}')
                        fig.add_annotation(x=row['left']+0.15*row['width'], y=row['bottom']+0.85*row['height'],text=index,font=dict(size=text_size+4,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['x']-0.15*row['width'], y=row['y']+0.25*row['height'],text=row['WFC'][:2],textangle=0,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x']-0.15*row['width'], y=row['y']-0.25*row['height'],text=row['WFC'][2:],textangle=0,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x']+0.25*row['width'], y=row['y'],text=row["all_defect"],textangle=0,font=dict(size=text_size,color='gray'),showarrow=False)

                elif 15>=row['width']>=8 and 10>row['height']>=3:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[2]})
                        # print(f'case {2}')
                        fig.add_annotation(x=row['left']+0.1*row['width'], y=row['top']-0.25*row['height'],text=index,font=dict(size=text_size+2,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y']+0.15*row['height'],text=row['WFC'],textangle=0,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y']-0.25*row['height'],text=row["all_defect"],textangle=0,font=dict(size=text_size,color='gray'),showarrow=False)
                elif 15>=row['width']>=8 and 3>row['height']:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[3]})
                        # print(f'case {2}')
                        fig.add_annotation(x=row['left']+0.12*row['width'], y=row['top']-0.25*row['height'],text=index,font=dict(size=text_size+1,color=index_color),showarrow=False)
                        fig.add_annotation(x=row['left']+0.16*row['width'], y=row['bottom']+0.25*row['height'],text=row['WFC'][:2],textangle=0,font=dict(size=text_size+1,color=text_color),showarrow=False)
                        fig.add_annotation(x=row['x']+0.1*row['width'], y=row['y']+0.1*row['height'],text=row['WFC'][2:],textangle=0,font=dict(size=text_size+1,color=text_color),showarrow=False)
                        # fig.add_annotation(x=row['x'], y=row['y']-0.35*row['height'],text=row["all_defect"],textangle=0,font=dict(size=text_size-1,color='gray'),showarrow=False)
                        # fig.add_annotation(x=row['x'], y=row['y']-0.35*row['height'],text=str(is_light_color(row["lgt_color_die"])),textangle=0,font=dict(size=text_size-1,color='gray'),showarrow=False)
                        
                elif 8>row['width']>=4 and 7>row['height']:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[4]})
                        fig.add_annotation(x=row['left']+0.15*row['width'], y=row['bottom']+0.85*row['height'],text=index,font=dict(size=text_size,color=index_color),showarrow=False)
                        fig.add_annotation(x=row['left']+0.7*row['width'], y=row['bottom']+0.8*row['height'],text=row['WFC'][:2],textangle=0,font=dict(size=text_size,color=text_color),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y']-0.15*row['height'],text=row['WFC'][2:],textangle=0,font=dict(size=text_size+1,color=text_color),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']-0.2*row['height'],text=row["all_defect"],textangle=0,font=dict(size=text_size,color='black'),showarrow=False)
                
                elif 4>row['width']>=3 and 5>row['height']:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[5]})
                        fig.add_annotation(x=row['x'], y=row['top']-1.2,text=index,font=dict(size=text_size+3,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['left']-0.5, y=row['y'],text=row['WFC'][:2],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']+1.8,text=row['WFC'][2:].split('D')[0],textangle=0,font=dict(size=text_size,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['right']+1, y=row['y'],text="D"+row['WFC'][2:].split('D')[1],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']-0.8,text=row["all_defect"],textangle=0,font=dict(size=text_size-1,color='gray'),showarrow=False)

                elif 3>row['width']>=1 and 5>row['height']:
                        # print(f'case {8}')
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[6]})
                        fig.add_annotation(x=row['x'], y=row['top']-1.2,text=index,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['left']-0.6, y=row['y'],text=row['WFC'][:2],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['top']+0.6,text=row['WFC'][2:].split('D')[0],textangle=0,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['right']+1, y=row['y'],text="D"+row['WFC'][2:].split('D')[1],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']-0.8,text=row["all_defect"],textangle=0,font=dict(size=text_size-1,color='gray'),showarrow=False)
                elif 1>row['width'] and 10>row['height']:
                        # print(f'case {8}')
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[7]})
                        fig.add_annotation(x=row['x'], y=row['y'],text=index,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['right']+1, y=row['top']-0.15*row['height'],text=row['WFC'][:2].split('D')[0],textangle=0,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['left']-1, y=row['y'],text=row['WFC'][2:],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        # fig.add_annotation(x=row['x'], y=row['top']+0.6,text=row['WFC'][2:].split('D')[0],textangle=0,font=dict(size=text_size,color='white'),showarrow=False)
                        # fig.add_annotation(x=row['right']+1, y=row['y'],text="D"+row['WFC'][2:].split('D')[1],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['right']+1, y=row['y'],text=row["all_defect"],textangle=-90,font=dict(size=text_size-1,color='gray'),showarrow=False)
                else:
                        # print(f'case {4}')
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[8]})
                        fig.add_annotation(x=row['left']+0.15*row['width'], y=row['bottom']+0.85*row['height'],text=index,font=dict(size=text_size+4,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['left']+0.5*row['width'], y=row['bottom']+0.8*row['height'],text=row['W'],textangle=0,font=dict(size=text_size+4,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']+0.5*row['height'],text="F"+str(row['F'])+row['D'],textangle=0,font=dict(size=text_size+4,color=row['lgt_gray_text']),showarrow=False)
                        # fig.add_annotation(x=row['left']+0.35*row['width'], y=row['bottom']+0.5*row['height'],text='MPL [dB/m]:',textangle=0,font=dict(size=text_size+2,color="#5A5C5A"),showarrow=False)
                        # fig.add_annotation(x=row['left']+0.72*row['width'], y=row['bottom']+0.5*row['height'],text=str(round(row['max_prop_loss [dB/m]'],3)),textangle=0,font=dict(size=text_size+3,color="#4ac859"),showarrow=False)
                        
                        fig.add_annotation(x=row['x'], y=row['bottom']+0.1*row['height'],text=row["all_defect"],textangle=0,font=dict(size=text_size,color='gray'),showarrow=False)
        
        # fig.add_annotation(x=delta_x, y=50,text='MPL [dB/m]: max_prop_loss [dB/m]',font=dict(size=11,color='gray'),showarrow=False,xanchor="left", align="left")
        y_index_ini=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
        # y_index_ini=[12,11,10,9,8,7,6,5,4,3,2,1]
        # x_index=[]
        y_index=[]
        # for xi in range(box_data['x'].unique().shape[0]):
        #         x_index.append(x_index_ini[xi])

        x_index=abs(np.arange(box_data['x'].unique().shape[0])+1).tolist()
   
        for yi in range(box_data['y'].unique().shape[0]):
                y_index.append(y_index_ini[yi])
       

        x_index_df=pd.DataFrame({'x':box_data['x'].unique(),'y':np.ones(box_data['x'].unique().shape[0]),'x_index':x_index})
        y_index_df=pd.DataFrame({'x':np.ones(box_data['y'].unique().shape[0]),'y':box_data['y'].unique(),'y_index':y_index})


        for index, row in x_index_df.iterrows():
                fig.add_annotation(x=row['x'], y=1,text=str(int(row['x_index'])),font=dict(size=18,color='white'),showarrow=False)
        for index, row in y_index_df.iterrows():
               fig.add_annotation(x=1, y=row['y'],text=row['y_index'],font=dict(size=18,color='white'),showarrow=False)
               
        # pre-binning
        user_google_drive=User_Info.get_user_GoogleDrive()
        if save_fig:
                if QR:
                        os.chdir(r''+user_google_drive+'/Engineering - Lots/'+lot_id+'/Quality reports')
                        fig.write_image(box_id+"-QR_2.png",format='png',width=1183,height=900,scale=3)
                else:
                        os.chdir(r''+ user_google_drive +'/Engineering - Lots/'+lot_id+'/Box maps')
                        fig.write_image(box_id+"-BM.png",format='png',width=1183,height=900,scale=3)

        return (fig,display_case_df)
        # return fig    
    
class Small_Box_Die_Display():
       @staticmethod
       def diplay_die_in_small_box(box_data,box_id, lot_id, defect_table_df, save_fig=True,QR=True):
        text_size=14
        delta_x=57.5

        fig = go.Figure()

        wafer_list=box_data['W_id'].unique().tolist()
        print(wafer_list)
        Wafer_df=pd.DataFrame({'text':wafer_list,
                                'x':delta_x*np.ones(len(wafer_list)),
                                'y':[25.5-i*1.42 for i in range(len(wafer_list))]})
        print(Wafer_df)
        many_defetcs=False
        fig=Small_Box.lgt_small_box_generator(fig=fig,box_id=box_id,lgt_oa_id=lot_id,defect_observed=not defect_table_df.empty,visual_axis=False,many_defetcs=many_defetcs)

        for index, row in Wafer_df.iterrows():
                fig.add_annotation(x=row['x'], y=row['y'],text=row['text'],font=dict(size=16,color='gray'),showarrow=False,xanchor="left", align="left")

        if not defect_table_df.empty:
                if not many_defetcs:
                        for index, row in defect_table_df.iterrows():
                                fig.add_annotation(x=row['xs'], y=row['ys'],text=row['defect'],font=dict(size=16,color='gray'),showarrow=False,xanchor="left", align="left")
                else:
                        for index, row in defect_table_df.iterrows():
                                fig.add_annotation(x=row['xs'], y=row['ys']+6.2,text=row['defect'],font=dict(size=16,color='gray'),showarrow=False,xanchor="left", align="left")             
        display_case_df=pd.DataFrame([])
        for index, row in box_data.iterrows():
                fig.add_trace(go.Scatter(x=row[['left','left','right','right','left']].values, 
                                y=row[['bottom','top','top','bottom','bottom']].values,
                                fill='toself',
                                # fillcolor = 'gold',
                                fillcolor = row['lgt_color_die'],
                                hoveron='points',
                                line_color=row['lgt_color_die'],
                                marker=dict(size=0, color='gold', opacity=0),showlegend=False))
                                #line_color='gold',marker=dict(size=0, color='goldenrod', opacity=0.2),showlegend=False))
                
                if row['width']>15 and 6>row['height']:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[1]})
                        # print(f'case {1}')
                        fig.add_annotation(x=row['left']+0.15*row['width'], y=row['bottom']+0.85*row['height'],text=index,font=dict(size=text_size+4,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['x']-0.15*row['width'], y=row['y']+0.25*row['height'],text=row['WFC'][:2],textangle=0,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x']-0.15*row['width'], y=row['y']-0.25*row['height'],text=row['WFC'][2:],textangle=0,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x']+0.25*row['width'], y=row['y'],text=row["all_defect"],textangle=0,font=dict(size=text_size,color='gray'),showarrow=False)

                elif 15>=row['width']>=8 and 10>row['height']>=3:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[2]})
                        # print(f'case {2}')
                        fig.add_annotation(x=row['left']+0.1*row['width'], y=row['top']-0.25*row['height'],text=index,font=dict(size=text_size+2,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y']+0.15*row['height'],text=row['WFC'],textangle=0,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        # fig.add_annotation(x=row['x'], y=row['y']-0.25*row['height'],text=row["all_defect"],textangle=0,font=dict(size=text_size,color='gray'),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y']-0.15*row['height'],text=row['all_defect'],textangle=0,font=dict(size=text_size-3,color='gray'),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y']-0.35*row['height'],text=row['C_defect'],textangle=0,font=dict(size=text_size-3,color='gray'),showarrow=False)
                elif 15>=row['width']>=8 and 3>row['height']:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[3]})
                        # print(f'case {2}')
                        fig.add_annotation(x=row['left']+0.1*row['width'], y=row['top']-0.25*row['height'],text=index,font=dict(size=text_size+2,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y']+0.15*row['height'],text=row['WFC'],textangle=0,font=dict(size=text_size+1,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y']-0.35*row['height'],text=row["all_defect"],textangle=0,font=dict(size=text_size-1,color='gray'),showarrow=False)
                elif 8>row['width']>=4 and 7>row['height']>=5:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[4]})
                        fig.add_annotation(x=row['left']+0.15*row['width'], y=row['bottom']+0.85*row['height'],text=index,font=dict(size=text_size,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['left']+0.7*row['width'], y=row['bottom']+0.8*row['height'],text=row['WFC'][:2],textangle=0,font=dict(size=text_size,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y'],text=row['WFC'][2:],textangle=0,font=dict(size=text_size+1,color=row['lgt_gray_text']),showarrow=False)

                        fig.add_annotation(x=row['x'], y=row['bottom']+0.25*row['height'],text=row["all_defect"],textangle=0,font=dict(size=text_size-4,color='white'),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']+0.05*row['height'],text=row["C_defect"],textangle=0,font=dict(size=text_size-4,color='white'),showarrow=False)

                elif 8>row['width']>=4 and 5>row['height']>1:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[5]})
                        fig.add_annotation(x=row['left']+0.15*row['width'], y=row['bottom']+0.8*row['height'],text=index,font=dict(size=text_size,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['left']+0.7*row['width'], y=row['bottom']+0.8*row['height'],text=row['WFC'][:2],textangle=0,font=dict(size=text_size,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['y']-0.05*row['height'],text=row['WFC'][2:],textangle=0,font=dict(size=text_size+1,color=row['lgt_gray_text']),showarrow=False)

                        fig.add_annotation(x=row['x'], y=row['bottom']+0.15*row['height'],text=row["all_defect"],textangle=0,font=dict(size=text_size-4,color='white'),showarrow=False)
                        # fig.add_annotation(x=row['x'], y=row['bottom']-0.55*row['height'],text=row["C_defect"],textangle=0,font=dict(size=text_size-4,color='white'),showarrow=False)
                        	
                
                elif 4>row['width']>=3 and 5>row['height']:
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[6]})
                        fig.add_annotation(x=row['x'], y=row['top']-1.2,text=index,font=dict(size=text_size+3,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['left']-0.5, y=row['y'],text=row['WFC'][:2],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']+1.8,text=row['WFC'][2:].split('D')[0],textangle=0,font=dict(size=text_size,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['right']+1, y=row['y'],text="D"+row['WFC'][2:].split('D')[1],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']-0.8,text=row["all_defect"],textangle=0,font=dict(size=text_size-1,color='gray'),showarrow=False)

                elif 3>row['width']>=1 and 5>row['height']:
                        # print(f'case {8}')
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[7]})
                        fig.add_annotation(x=row['x'], y=row['top']-1.2,text=index,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['left']-0.6, y=row['y'],text=row['WFC'][:2],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['top']+0.6,text=row['WFC'][2:].split('D')[0],textangle=0,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['right']+1, y=row['y'],text="D"+row['WFC'][2:].split('D')[1],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']-0.8,text=row["all_defect"],textangle=0,font=dict(size=text_size-1,color='gray'),showarrow=False)

                elif 1>row['width'] and 10>row['height']:
                        # print(f'case {8}')
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[8]})
                        fig.add_annotation(x=row['x'], y=row['y'],text=index,font=dict(size=text_size+2,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['right']+1, y=row['top']-0.15*row['height'],text=row['WFC'][:2].split('D')[0],textangle=0,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['left']-1, y=row['y'],text=row['WFC'][2:],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        # fig.add_annotation(x=row['x'], y=row['top']+0.6,text=row['WFC'][2:].split('D')[0],textangle=0,font=dict(size=text_size,color='white'),showarrow=False)
                        # fig.add_annotation(x=row['right']+1, y=row['y'],text="D"+row['WFC'][2:].split('D')[1],textangle=-90,font=dict(size=text_size,color='white'),showarrow=False)
                        fig.add_annotation(x=row['right']+1, y=row['y'],text=row["all_defect"],textangle=-90,font=dict(size=text_size-1,color='gray'),showarrow=False)
                else:
                        # print(f'case {4}')
                        display_case_df=LGT_df_General.append_row(display_case_df,{'index':[index],'width':[row['width']],'height':[row['height']],'case':[9]})
                        fig.add_annotation(x=row['left']+0.15*row['width'], y=row['bottom']+0.85*row['height'],text=index,font=dict(size=text_size,color=row['lgt_color_text']),showarrow=False)
                        fig.add_annotation(x=row['left']+0.7*row['width'], y=row['bottom']+0.8*row['height'],text=row['WFC'][:2],textangle=0,font=dict(size=text_size,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']+0.25*row['height'],text=row['WFC'][2:],textangle=0,font=dict(size=text_size+1,color=row['lgt_gray_text']),showarrow=False)
                        fig.add_annotation(x=row['x'], y=row['bottom']+0.15*row['height'],text=row["all_defect"],textangle=0,font=dict(size=text_size,color='gray'),showarrow=False)        

                # fig.add_annotation(x=row['x']+0.2, y=row['y'],text=row['WFC'],textangle=-90,font=dict(size=12,color='gray'),showarrow=False)
                # fig.add_annotation(x=row['x']+1.2, y=row['y'],text=row["all_defect"],textangle=-90,font=dict(size=12,color='gray'),showarrow=False)
                # fig.add_trace(go.Scatter(x=[row['x']], y=[row['y']],mode='markers',marker=dict(size=4,color='red', opacity=0.5),showlegend=False))

        # pre-binning
        user_google_drive=User_Info.get_user_GoogleDrive()
        if save_fig:
                if QR:
                        os.chdir(r''+user_google_drive+'/Engineering - Lots/'+lot_id+'/Quality reports')
                        fig.write_image(box_id+"-QR_4.png",format='png',width=1250,height=720,scale=3)
                else:
                        os.chdir(r''+ user_google_drive +'/Engineering - Lots/'+lot_id+'/Box maps')
                        fig.write_image(box_id+"-BM.png",format='png',width=1250,height=720,scale=3)

        return (fig,display_case_df)
   