from sklearn.decomposition import PCA
import pandas as pd


def pca(dataframe_without_target, variance_explained):
    """
    Gets new dataframe that has been transformed by PCA
    :param dataframe_without_target: pandas.DataFrame
    :param variance_explained: float from 0 to 1
    :return: pandas.DataFrame
    """
    pca_model = PCA(svd_solver='full', n_components=variance_explained)
    pca_model.fit(dataframe_without_target)
    print("num components: {0}".format(len(pca_model.components_)))
    print("feature vector: {0}".format(pca_model.components_))
    dataframe_pca = pd.DataFrame(pca_model.transform(dataframe_without_target))
    return dataframe_pca
