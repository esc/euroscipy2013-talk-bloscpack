import pylab as plt
plt.rcParams.update({'font.size': 10})

def result_filter(storage, size, codecs):
    # consider only ssd storage
    it = df[df['storage'] == storage]
    # consider only large test size
    it = it[it['size'] == size]
    # compare bloscpack with npy
    it = it[it['codec'].isin(codecs)]
    idx = list(it.columns)
    idx.pop(1) # remove the storage column
    idx.pop(-1) # remove the ratio column
    it = it[idx]
    # reindex using entropy, codec and level
    it = it.set_index(['entropy', 'codec', 'level'])
    return it

def plot_codec_comp( storage, codecs):
    # filter out the results
    it_small = result_filter(storage, 'small', codecs)
    it_mid = result_filter(storage, 'mid', codecs)
    it_large = result_filter(storage, 'large', codecs)
    # set up the subplots
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15,15), sharey=True)
    # configure the keyword arguments
    kwargs = {'legend': None,
              'kind': 'barh',
              'colormap': 'gray',
              }
    # set the xlimits for each column
    kwargs_small = kwargs.copy()
    kwargs_small['xlim'] = (0, it_small.max()[1:].max())
    kwargs_mid = kwargs.copy()
    kwargs_mid['xlim'] = (0, it_mid.max()[1:].max())
    kwargs_large = kwargs.copy()
    kwargs_large['xlim'] = (0, it_large.max()[1:].max())
    it_large.max()[1:].max()
    # do plotting
    for i, ent in enumerate(['low', 'medium', 'high']):
        it_small.ix[ent].plot(ax=ax[i, 0], **kwargs_small)
        ax[i, 0].set_title('small size - %s entropy' % ent)
        it_mid.ix[ent].plot(ax=ax[i, 1], **kwargs_mid)
        ax[i, 1].set_title('medium size - %s entropy' % ent)
        if ent == 'low':
            kwargs_copy = kwargs_large.copy()
            kwargs_copy['legend'] = True
            it_large.ix[ent].plot(ax=ax[i, 2], **kwargs_copy)
        else:
            it_large.ix[ent].plot(ax=ax[i, 2], **kwargs_large)
        ax[i, 2].set_title('large size - %s entropy' % ent)

def plot_ratio(size, entropy, storage='ssd', ax=None):
    it = df
    it = it[it['storage'] == 'ssd']
    it = it[it['size'] == size]
    it = it[it['entropy'] == entropy]
    it = it[['level', 'codec', 'ratio']]
    it = it.set_index(['codec', 'level'])
    it.plot(kind='barh', figsize=(15, 8), ax=ax, legend=False,
            title="size: '%s' entropy: '%s'" % (size, entropy), colormap='gray')

