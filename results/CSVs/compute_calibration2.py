import pandas as pd
import sys

try:
    mc = pd.read_csv('xai_mc_dropout_metrics.csv')
    seg = pd.read_csv('summary_SegResNet_metrics.csv')

    # Clean seg patient names to match mc
    seg['Patient'] = seg['Patient'].apply(lambda x: x if '-000' in x else f'{x}-000')

    merged = pd.merge(mc, seg, on=['Patient', 'Region'], suffixes=('_mc', '_seg'))
    
    with open('mc_calibration.txt', 'w') as f:
        for region in ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor']:
            df = merged[merged['Region'] == region].copy()
            if len(df) > 0:
                df['Error'] = 1.0 - df['Dice']
                corr = df['UAR'].corr(df['Error'])
                f.write(f'{region:16s}: Pearson r(UAR, Error) = {corr:.3f}\n')
            else:
                f.write(f'{region:16s}: No data to compute correlation\n')
except Exception as e:
    with open('mc_calibration.txt', 'w') as f:
        f.write(f'Error: {e}\n')
