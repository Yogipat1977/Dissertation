import pandas as pd
mc = pd.read_csv('xai_mc_dropout_metrics.csv')
seg = pd.read_csv('summary_SegResNet_metrics.csv')

# Clean seg patient names to match mc
# e.g. BraTS-GLI-00022-000
seg['Patient'] = seg['Patient'].apply(lambda x: x if '-000' in x else f'{x}-000')

merged = pd.merge(mc, seg, on=['Patient', 'Region'], suffixes=('_mc', '_seg'))
for region in ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor']:
    df = merged[merged['Region'] == region].copy()
    df['Error'] = 1.0 - df['Dice']
    corr = df['UAR'].corr(df['Error'])
    print(f'{region:16s}: Pearson r(UAR, Error) = {corr:.3f}')
