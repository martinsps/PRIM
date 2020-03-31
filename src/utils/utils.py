
def data_frame_difference(df1, df2):
    """
    Returns difference of two data frames, that is, the elements
    that are in df1 but not in df2.
    :param df1: First data frame
    :param df2: Second data frame
    :return: The difference of df1 and df2
    """
    return df1[~df1.apply(tuple, 1).isin(df2.apply(tuple, 1))]
