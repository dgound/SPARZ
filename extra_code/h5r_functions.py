#%%

def h5r_to_df(filepath):
    '''
    This function takes a filepath to an hdf5 file and returns a pandas dataframe
    '''
    import h5py
    import numpy as np
    import pandas as pd

    # open an hdf5 file in read mode
    f = h5py.File(filepath, 'r')

    dataset = f["FitResults"]
    data = dataset[:]

    fresultdtype = [('tIndex', '<i4'),
                ('fitResults', [('A', '<f4'), ('x0', '<f4'), ('y0', '<f4'), ('z0', '<f4'), ('bg', '<f4'), ('br', '<f4'), ('dx', '<f4'), ('dy', '<f4')]),
                ('fitError', [('A', '<f4'), ('x0', '<f4'), ('y0', '<f4'), ('z0', '<f4'), ('bg', '<f4'), ('br', '<f4'), ('dx', '<f4'), ('dy', '<f4')]),
                ('startParams', [('A', '<f4'), ('x0', '<f4'), ('y0', '<f4'), ('z0', '<f4'), ('bg', '<f4'), ('br', '<f4'), ('dx', '<f4'), ('dy', '<f4')]),
                ('resultCode', '<i4'),
                ('slicesUsed', [('x', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')]),
                                ('y', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')]),
                                ('x2', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')]),
                                ('y2', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')])]),
                ('subtractedBackground', [('g', '<f4'), ('r', '<f4')]),
                ('ratio', '<f4'),
                ('nchi2', '<f4')]
    
    df = pd.DataFrame(data, columns=[x[0] for x in fresultdtype])

    # Extract nested tuples and create separate columns
    df_fit_results = pd.DataFrame(df['fitResults'].tolist(), columns=['fitResults_' + x for x in ['A', 'x0', 'y0', 'z0', 'bg', 'br', 'dx', 'dy']])
    df_fit_error = pd.DataFrame(df['fitError'].tolist(), columns=['fitError_' + x for x in ['A', 'x0', 'y0', 'z0', 'bg', 'br', 'dx', 'dy']])
    df_start_params = pd.DataFrame(df['startParams'].tolist(), columns=['startParams_' + x for x in ['A', 'x0', 'y0', 'z0', 'bg', 'br', 'dx', 'dy']])
    df_slices_used_x = pd.DataFrame(df['slicesUsed'].apply(lambda x: x[0]).tolist(), columns=['slicesUsed_x_' + x for x in ['start', 'stop', 'step']])
    df_slices_used_y = pd.DataFrame(df['slicesUsed'].apply(lambda x: x[1]).tolist(), columns=['slicesUsed_y_' + x for x in ['start', 'stop', 'step']])
    df_slices_used_x2 = pd.DataFrame(df['slicesUsed'].apply(lambda x: x[2]).tolist(), columns=['slicesUsed_x2_' + x for x in ['start', 'stop', 'step']])
    df_slices_used_y2 = pd.DataFrame(df['slicesUsed'].apply(lambda x: x[3]).tolist(), columns=['slicesUsed_y2_' + x for x in ['start', 'stop', 'step']])

    # Concatenate the extracted columns with the original DataFrame
    df = pd.concat([df,
                    df_fit_results, df_fit_error, df_start_params,
                    df_slices_used_x, df_slices_used_y, df_slices_used_x2, df_slices_used_y2], axis=1)


    # for column in df.columns:
    #     column_type = df[column].apply(type).iloc[0]
        # print(f"Column '{column}' has type: {column_type}")

    # Convert the 'fitResults', 'fitError', and 'startParams' columns to strings
    df['fitResults'] = df['fitResults'].astype(str)
    df['fitError'] = df['fitError'].astype(str)
    df['startParams'] = df['startParams'].astype(str)
    df['slicesUsed'] = df['startParams'].astype(str)
    df['subtractedBackground'] = df['startParams'].astype(str)

    # for column in df.columns:
    #     column_type = df[column].apply(type).iloc[0]
        # print(f"Column '{column}' has type: {column_type}")

    df = df.drop(columns=['fitResults', 'fitError', 'startParams', 'slicesUsed'])


    return df




