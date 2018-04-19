# -*- coding: utf-8 -*-
"""
@author: fklimenka
"""
from __future__ import print_function, division
import os
import pandas as pd
import numpy as np
import tables as pt
from pandas import DataFrame
import time
import datetime as dt
from datetime import datetime
from pandas.tools.plotting import autocorrelation_plot, scatter_matrix
import scipy.stats
        

def get_dates(filename):

    f = pt.openFile(filename+'.h5','r')
    year_groups = f.root._v_groups
    data = {}
     
    for year_key in year_groups.viewkeys():
        month_groups = year_groups[year_key]
        for month_group in month_groups:
          for day_group in month_group:
              attrs = day_group._v_attrs
              dict_key = year_key + '-' + month_group._v_name + '-' + day_group._v_name
              data[dict_key] = [attrs['QuoteStart'],
                                attrs['QuoteCount'],
                                attrs['TradeStart'],
                                attrs['TradeCount']]

    return data

LIQUID_HOURS={'GC':['07:20','12:30'],
             'SI':['07:25','12:25'],
             'CL':['08:00','13:30'],
             'NG':['08:00','13:30'],
             'C':['08:30','13:15'],
             'S':['08:30','13:15'],
             'W':['08:30','13:15'],
             'ES':['08:30','15:15'],
             'YM':['08:30','15:15'],           
             'TY':['07:20','14:00'],
             'BP':['07:20','14:00'],
             'URO':['07:20','14:00']}    



path='C:/Users/fklimenka/futures_data/'


def get_cleaned_day(path, TICKER, contract, day, price_type='trades'):
    
    f = pt.openFile(path+TICKER+'/'+contract+'.h5','r')
    dates=get_dates(path+TICKER+'/'+contract)
    md = dates[day]
    
    if price_type=='trades':
        start = md[2]
        count = md[3]
        table = f.root.Trades
    elif price_type=='quotes':
        start = md[0]
        count = md[1]
        table=f.root.Quotes
    
    d=DataFrame(table.read(start,start+count))   
    
    
    if TICKER in ['GC','SI','CL','NG']:#,'C','S','W']:
        d.utcsec=d.utcsec+(d.offset-1)*3600       
    else:
        d.utcsec=d.utcsec+d.offset*3600   
    
    
    if price_type=='trades':
        #get rid of yesterday values: utcsec<0
        d=d[d.utcsec>0]
        #get rid of zero prices
        d=d[d.price>0]
        #price outliers deletion
        d=d[abs(d.price.pct_change())<0.1]
    elif price_type=='quotes':
        #get rid of yesterday values: utcsec<0
        d=d[d.utcsec>0]    
        #replace zero bids and asks with their previous values: i.e. ffill nans
        d.loc[d.bid==0,'bid']=np.nan
        d.bid=d.bid.ffill()
        d.loc[d.bidsize==0,'bidsize']=np.nan
        d.bidsize=d.bidsize.ffill()
        d.loc[d.ask==0,'ask']=np.nan
        d.ask=d.ask.ffill()
        d.loc[d.asksize==0,'asksize']=np.nan
        d.asksize=d.asksize.ffill()
        d=d.dropna()
    
    d_datetime=map(lambda x,y,z: dt.datetime.strptime(str(int(x))+time.strftime("%H:%M:%S", time.gmtime(int(y))).replace(":", "")+str(z/100).replace(".", ""),'%Y%m%d%H%M%S%f') ,
                       d.yyyymmdd, np.array(d.utcsec), d.ms)

        
    d.index=d_datetime    
    
    d=d.dropna().between_time(LIQUID_HOURS[TICKER][0], LIQUID_HOURS[TICKER][1])
    
    return d



def aggregate(data, price_type='trades'):
    if price_type=='trades':
        ag_data=pd.DataFrame()
        ag_data['price']=data['price'].groupby(data.index).median()
        ag_data['volume']=data['volume'].groupby(data.index).sum()
    elif price_type=='quotes':        
        ag_data=pd.DataFrame()
        ag_data['bid']=data['bid'].groupby(data.index).median()
        ag_data['ask']=data['ask'].groupby(data.index).median()
        ag_data['bidsize']=data['bidsize'].groupby(data.index).sum()
        ag_data['asksize']=data['asksize'].groupby(data.index).sum()
    return ag_data


  

"GOLD AND SILVER FUTURES"
day='2012-DEC-11'
'SILVER'
trades_SI=get_cleaned_day(path, 'SI', 'SI_H_2013', day, price_type='trades')
quotes_SI=get_cleaned_day(path, 'SI', 'SI_H_2013', day, price_type='quotes')
ag_trades_SI=aggregate(trades_SI, price_type='trades')
ag_quotes_SI=aggregate(quotes_SI, price_type='quotes')
DUR=pd.DataFrame(np.diff(ag_trades_SI.index.values)/np.timedelta64(1, 's'), columns=['duration'])
SPR_SI=np.log(ag_quotes_SI.ask)-np.log(ag_quotes_SI.bid)
DPTH_SI=np.log(ag_quotes_SI.asksize)-np.log(ag_quotes_SI.bidsize)
#cleaning: drop negative spreads
SPR_SI=SPR_SI[SPR_SI>=0]


'GOLD'
trades_GC=get_cleaned_day(path, 'GC', 'GC_G_2013', day, price_type='trades')
quotes_GC=get_cleaned_day(path, 'GC', 'GC_G_2013', day, price_type='quotes')
ag_trades_GC=aggregate(trades_GC, price_type='trades')
ag_quotes_GC=aggregate(quotes_GC, price_type='quotes')
SPR_GC=np.log(ag_quotes_GC.ask)-np.log(ag_quotes_GC.bid)
DPTH_GC=np.log(ag_quotes_GC.asksize)-np.log(ag_quotes_GC.bidsize)
#cleaning: drop negative spreads
SPR_GC=SPR_GC[SPR_GC>=0]


'DESCRIPTIVE PLOTS'
#for silver only
'TRADES: SI and GC'
#price
ax1=ag_trades_SI.price.plot(ax=plt.subplot(3,1,1, sharex=None), color='blue', legend=False)
ax2=plt.twinx()
ag_trades_GC.price.plot(color='red',legend=False, sharex=None)
plt.title('price: silver and gold')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, ['silver' ,'gold'], loc='upper right', frameon=False)
#volume
ag_trades_SI.volume.resample('10min',how='sum',label='right').plot(ax=plt.subplot(3,1,2), color='blue', legend=False, sharex=None)
ag_trades_GC.volume.resample('10min',how='sum',label='right').plot(color='red', legend=False, sharex=None)
plt.legend(['silver', 'gold'], loc='upper right', frameon=False)
plt.title('volume: silver and gold')
#duration
(DUR*1000).plot(ax=plt.subplot(3,1,3),kind='hist',color='k', bins=100, legend=False, sharex=None)
plt.text(20000,1000,str((DUR*1000).describe()))
plt.title('duration between aggregated trades (milliseconds) for silver')


'QUOTES: SI & GC'
#1.DEPTH PLOT
DPTH_SI.plot(color='k', ax=plt.subplot(3,2,1))
plt.title('depth: silver')
DPTH_GC.plot(color='k', ax=plt.subplot(3,2,2))
plt.title('depth: gold')
#2 histogram: depth
plt.subplot(3,2,3)
DPTH_SI.hist(bins=50)
plt.title('hist: depth: silver')
plt.text(-4.9, 1500, DPTH_SI.describe())
plt.subplot(3,2,4)
DPTH_GC.hist(bins=50)
plt.title('hist: depth: gold')
plt.text(-5.9, 1900, DPTH_GC.describe())
#5. autocorrelation: depth
autocorrelation_plot(DPTH_SI, ax=plt.subplot(3,2,5), color='k')
plt.title('autocorrelation: depth silver')
plt.xlim(0,70)
autocorrelation_plot(DPTH_GC, ax=plt.subplot(3,2,6), color='k')
plt.title('autocorrelation: depth gold')
plt.xlim(0,70)




'======================================================================================================='
'HAZARD RATE: multivariate'
hdf=pd.concat((ag_trades_SI['price'], DPTH_SI, DPTH_GC),
              axis=1, keys=['T','x1','w1'])

hdf[['x1','w1']]=hdf[['x1','w1']].ffill()
hdf=hdf.between_time('07:25', '12:25')

#indexing: looking at chnages in events
trades_index=~np.isnan(hdf['T'])
hdf['event_index']=np.nan
hdf['event_index'][trades_index]=range(1,len(matplotlib.mlab.find(trades_index))+1)
hdf['event_index']=hdf['event_index'].bfill()

#sub-indexing: looking at changes in covariates
hdf['sub_index']=np.nan
for i in set(hdf.event_index):
    l=(hdf[hdf['event_index']==i][['x1','w1']].diff()!=0).any(axis=1)
    l=l[l]
    hdf['sub_index'].ix[l.index]=range(1,len(l)+1)
    print(i)

hdf['sub_index']=hdf['sub_index'].ffill()
hdf['s_index']=hdf['event_index']+0.01*hdf['sub_index']

#time since last trade: s
hdf['s']=np.nan
for i in range(2, int(hdf.event_index.max())+1):
    h=hdf[hdf['event_index']==i]
    l=(hdf[hdf['event_index']==i].index-hdf[hdf['event_index']==i-1].index[-1])/np.timedelta64(1, 's')
    hdf['s'].ix[h.index]=l
    print(i)

hdf['time_index']=hdf.index
#duration of unchanged covariates
delta=(hdf.time_index.groupby(hdf.s_index).first().shift(-1)-hdf.time_index.groupby(hdf.s_index).first())/np.timedelta64(1, 's')

'normalising covariates by their std'
for c in ['x1','w1']:
    hdf[c]=hdf[c]/hdf[c].std()


#GRAPH
'SI vs GC'
#1. hexbin-plot: spread
hdf.plot(kind='scatter', x='x1', y='w1', ax=plt.subplot(1,1,1))
plt.axhline(y=0, color='blue', ls='--')
plt.axvline(x=0, color='blue', ls='--')
plt.title('scatter: depth')
plt.ylabel('depth of gold')
plt.xlabel('depth of silver')
plt.axhline(y=hdf['w1'].quantile(0.01), color='red', ls='-', lw=0.4)
plt.axhline(y=hdf['w1'].quantile(0.99), color='red', ls='-', lw=0.4)
plt.axvline(x=hdf['x1'].quantile(0.01), color='red', ls='-', lw=0.4)
plt.axvline(x=hdf['x1'].quantile(0.99), color='red', ls='-', lw=0.4)
plt.fill_between(np.linspace(hdf['x1'].quantile(0.01), hdf['x1'].quantile(0.99),2),
                 hdf['w1'].quantile(0.01), hdf['w1'].quantile(0.99), alpha=0.2, color='coral')



#lenght of each event
event_length=map(lambda x: len(matplotlib.mlab.find(hdf['event_index']==x)),sorted(set(hdf['event_index'].dropna())))
event_length=pd.DataFrame(event_length, columns=['#intra-trade quotes'])
event_length.hist(bins=69, edgecolor='none')
plt.text(100, 200, str(event_length.describe()))
#look at the number of times spread changes within an invent
quote_change=hdf['sub_index'].groupby(hdf['event_index']).last()
quote_change.plot(kind='hist', bins=55)
plt.text(15,500, quote_change.describe())



def kernel_alpha(hdf, b, p=np.array([hdf['x1'].mean(),hdf['w1'].mean()])):
    "kernel specified in Wolter (2014)"
    u=(np.array([p[0], p[1]])-hdf[['x1','w1']])/b   
    zeros=pd.DataFrame(data=np.zeros(len(u)), index=u.index)
    k=((np.maximum(1-u**2, zeros)**4.1)*((9/8-(15/8)*(u**2))+(9/2)*(144/(384**2))*(1680*(u**4)-1440*(u**2)+144))/b).product(axis=1)
    num=k.groupby(hdf.event_index).last().sum()
    denom=(k.groupby(hdf.s_index).last()*delta).sum()
    a=num/denom
    return a




'======================================================================================================='
titles={'x1':'DEPTH_SI', 'w1':'DEPTH_GC'}

def hazard_plot(j, b, hdf):
    x1=hdf.x1.mean()
    w1=hdf.w1.mean()
    alphas=[]    
    for k in np.linspace(hdf[j].quantile(0.001), hdf[j].quantile(0.999), 10):
        p=[x1,w1]
        if j[0]=='x':
            p[int(j[-1])-1]=k
        elif j[0]=='w':
            p[int(j[-1])+1]=k
        alphas.append([k, kernel_alpha(hdf, b, p=[p[0], p[1]])])
        print(p)
    
    df=pd.DataFrame(alphas)
    df.index=df[0]
    df[1].plot()
    df[1].plot(style='o')
    
    plt.axhline(y=kernel_alpha(hdf, b, [hdf.x1.mean(),hdf.w1.mean()]))
    plt.title(j+': '+titles[j])
    plt.xlabel(j)
    plt.ylabel('hazard rate')
        
    return alphas
        

hazard_plot('x1', 30, hdf)


'PRELIMINARY PLOTS'
count=0
for c in ['x1','w1']:
    count=count+1
    plt.subplot(2,2,count)
    hazard_plot(c, 4, hdf)
    

'AVERAGING'
count=0
C=alpha_df.alpha.mean()
for j in ['x1','w1']:
    count=count+1
    alpha_one=(alpha_df.alpha.groupby(alpha_df[j]).mean())*(1/C)
    alpha_one.plot(ax=plt.subplot(2,1,count), sharex=None)
    alpha_one.plot(style='o',ax=plt.subplot(2,1,count), sharex=None)
    plt.title(j+': '+titles[j])
    plt.xlabel(titles[j])
    plt.ylabel('Q(b)')
    plt.axhline(y=C)
    
'3-D plot'
ax=plt.subplot(1,1,1, projection='3d')
ax.plot_trisurf(alpha_df['x1'], alpha_df['w1'], alpha_df['alpha'], cmap=cm.jet, linewidth=0.2)
ax.set_xlabel('silver depth')
ax.set_ylabel('gold depth')
ax.set_zlabel(r'$ \hat{\alpha} $')
plt.title('hazard rate')

alpha_df['C']=C
ax.plot_trisurf(alpha_df['x1'], alpha_df['w1'], alpha_df['C'], cmap=cm.jet, alpha=0.1, linewidth=0.1)
ax.set_xlabel('silver depth')

'--------------------------------------------------------------------------------------------------------------------'
'Cross-validation'
#creating a matrix of alphas to be used in next stage
b=30

Q_b=[]

for b in np.linspace(1.65, 1.8, 21):
    print(b)
    '1st element'
    #alpha_CV on the grid
    alpha_CV=[]
    count=0
    for i in np.linspace(hdf['x1'].quantile(0.01), hdf['x1'].quantile(0.99), 20):
        for j in np.linspace(hdf['w1'].quantile(0.01), hdf['w1'].quantile(0.99), 20):
            alpha_CV.append([i, j, kernel_alpha(hdf, b, p=[i, j])])
            count=count+1
            print(count)
        
    alpha_df=pd.DataFrame(alpha_CV, columns=['x1', 'w1', 'alpha'])

    'evaluation of 1: binning'
    def binning(hdf, c, q=[0.01, 0.99]):
        bins=np.linspace(hdf[c].quantile(q[0]), hdf[c].quantile(q[1]), 20)
        digitized=np.digitize(hdf[c], bins)
        digitized[digitized==len(bins)]=len(bins)-1
        return bins[digitized]

    'binning'
    'making hdf values discrete'
    hdf_binned=pd.DataFrame()
    hdf_binned['x1']=binning(hdf, 'x1', q=[0.01, 0.99])
    hdf_binned['w1']=binning(hdf, 'w1', q=[0.01, 0.99])


    'need to create hdf binned'
    'calculating alpha out of these discrtized hdf_binned: we already have the values as we did it on the grid'
    hdf_binned_alpha=pd.DataFrame()
    count=0
    for p in hdf_binned[['x1','w1']].values:
        count=count+1
        hdf_binned_alpha=hdf_binned_alpha.append(alpha_df[(alpha_df['x1']==p[0]) & \
                                                          (alpha_df['w1']==p[1])])
        print(count)
        
    hdf_binned_alpha.index=hdf.index
    
    term_1=((hdf_binned_alpha.alpha**2).groupby(hdf.s_index).last()*delta).sum()
    
    '2nd element'
    #cannot apply the previous grid since these objects involve excluding eveint i when computing alphs
    alpha_i_CV=[]
    for i in sorted(set(hdf.event_index.dropna())):
        p=hdf.groupby(hdf.event_index).last().ix[i]
        alpha_i_CV.append([i, kernel_alpha(hdf[hdf.event_index!=i], b, p=[p['x1'], p['w1']])])    
        print(i)
        
    alpha_i_CV=pd.DataFrame(alpha_i_CV, columns=['event_index', 'alpha_i'])
    
    term_2=-2*alpha_i_CV.alpha_i.sum()
    
    '3rd element'
    def l_j(hdf, b, p=np.array([hdf['x1'].mean(),hdf['w1'].mean()])):
        "kernel specified in Wolter (2014)"
        u=(np.array([p[0], p[1]])-hdf[['x1','w1']])/b   
        zeros=pd.DataFrame(data=np.zeros(len(u)), index=u.index)
        
        k=(((np.maximum(1-u**2, zeros)**4.1)*((9/8-(15/8)*(u**2))+(9/2)*(144/(384**2))*(1680*(u**4)-1440*(u**2)+144)))/b).product(axis=1)
        num=(9/8)+(9/2)*((144**2)/(384**2))*(1/b)#k.groupby(hdf.event_index).last().sum()' MAKE IT 1/b k(0)
        denom=(k.groupby(hdf.s_index).last()*delta).sum()
        
        lj=num/denom
        return lj    
    
    H_b_i=[]
    for i in sorted(set(hdf.event_index.dropna())):
        p=hdf.groupby(hdf.event_index).last().ix[i]
        H_b_i.append([i, l_j(hdf, b, p=[p['x1'], p['w1']])])    
        print(i)
        
    H_b_i=pd.DataFrame(H_b_i, columns=['event_index', 'l_j'])
    H_b=H_b_i.l_j.sum()
    n=sorted(set(hdf.event_index.dropna()))[-1]
    term_3=(2*(H_b+1))/(n-H_b-2)
   
    'Cross-validation Q value'
    'should be 1/n here'
    Q=(((hdf_binned_alpha.alpha**2).groupby(hdf.s_index).last()*delta).sum() \
        -2*alpha_i_CV.alpha_i.sum())*(1/n)+term_3
      
    Q_b.append([b, Q, term_3])

dt.datetime.now()  
pd.DataFrame(Q_b).to_pickle('Q_b_depth_11DEC_8_new_b')


'--------------------------------------------------------------------------------------------------------------------'
ax1=ag_trades_SI.price.plot(ax=plt.subplot(3,1,1, sharex=None), color='blue', legend=False)
ax2=plt.twinx()
ag_trades_GC.price.plot(color='red',legend=False, sharex=None)
plt.subplots_adjust(bottom=0.1)
plt.title('price: silver and gold')




