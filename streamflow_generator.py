import numpy as np
import pandas as pd
import warnings
import tables

from scipy.linalg import cholesky


class StreamFlowsGeneration:
    """
    Python implementation of a synthetic streamflow generator based on Cholesky decomposition.
    Method developed by Kirsch et al. (2013)

    This class is inspired by the Matlab implementation found in https://github.com/jdherman/Qsynth and
    https://github.com/julianneq/Kirsch-Nowak_Streamflow_Generator

    Parameters:
    -----------
    dataframe: DataFrame
                A pandas DataFrame containing the historical inflows for the different locations in the basing.
                Each location must to be in a separately column in the DataFrame.
                **Note**: be sure you remove the index column and any `str` column name.

    monthly: bool, default False
            If True your historical inflows are in a monthly timestep.

    weekly: bool, default False
            If True your historical inflows are in a weekly timestep.

    daily: bool, default False
            If True your historical inflows are in a daily timestep.

    sample_size: integer, default None
                Number of sample to generate by each inflows locations.

    **Citations**:

    * Kirsch, B. R., G. W. Characklis, and H. B. Zeff (2013), Evaluating the impact of alternative hydro-climate scenarios on transfer agreements:
      Practical improvement for generating synthetic streamflows, Journal of Water Resources Planning and Management, 139(4), 396–406.

    * Herman, J.D, H.B. Zeff, J.R. Lamontagne, P.M. Reed, and G.W. Characklis, Synthetic drought scenario generation to support bottom-up
      water supply vulnerability assessments, Journal of Water Resources Planning and Management, 142(11), 04016050, 2016.
    """

    def __init__(self, dataframe, monthly=False, weekly=False, daily=False, sample_size=None):

        self.Qh = dataframe
        self.monthly = monthly
        self.weekly = weekly
        self.daily = daily
        self.sample = sample_size

        # _, refer to internal variables
        self._Log_Qhmean = []
        self._Log_Qhstd = []
        self._ncol = 0

        # defining the number of columns to reorder the historical data
        if self.monthly is True:
            self._ncol = 12

        if self.weekly is True:
            self._ncol = 52

        if self.daily is True:
            self._ncol = 365

    def qh_shifting(self, new_dataframe=None):
        """
        create a shifted dataframe based on the historical inflows. This is latter used to to maintain the interannual
        correlation information from the historical inflows. see Kirsch et al. (2013).

        Parameters:
        -----------
        new_dataframe: DataFrame, optional.
                        A pandas DataFrame containing the historical inflows for the different locations in the basing.
                        Each location must to be in a separately column in the DataFrame.
                        **Note**: be sure you remove the index column and any `str` column name.

        Returns:
        --------
        Qh_shifted: shifted dataframe, this is latter used to calculate the interannual correlation information in the
                    historical inflows.
        """

        # all the methods have a new_dataframe parameter. This is used if the user want to go step by step and create
        # the synthetic sample. I don't know if this is the best way to do this. probably not! :D
        if new_dataframe is None:
            tem = self.Qh
        else:
            tem = new_dataframe

        Qh_shifted = []

        if self.monthly is True or self.weekly is True:
            div = int(self._ncol / 2)
            Qh_shifted = tem[div:-div]

        if self.daily is True:
            div = int(np.floor(self._ncol / 2))
            Qh_shifted = tem[div:-div - 1]

        return Qh_shifted

    def formatting_qh(self, shifted=None, new_dataframe=None):
        """
        create the `Y_(i,j)` matrix in Kirsch et al. (2013). This matrix contain the historical inflows.
        where each row contain a single year's data. `i` is the number of years of information in the historical inflows
        and `j` take a value of 12, 52 or 365 according to the data timestep.

        Parameters:
        -----------
        new_dataframe: DataFrame, optional.
                        A pandas DataFrame containing the historical inflows for the different locations in the basing.
                        Each location must to be in a separately column in the DataFrame.
                        **Note**: be sure you remove the index column and any `str` column name.

        shifted: bool, default False
                If True, you are calling the `self.qh_shifting()` method internally.

        Returns:
        --------
        Qh_matrix: a reordered matrix containing the historical inflows data.
                   This formatted data is necessary for the methodology.
        """

        if new_dataframe is None:
            tem = self.Qh

        else:
            tem = new_dataframe

        self._Qhmean = []
        self._Qhstd = []
        series = []

        if shifted is False:
            series = tem

        if shifted is True:
            series = self.qh_shifting(tem)

        j_len = series.shape[1]
        i_len = series.shape[0]

        Qh_matrix = []
        Qh_tem = []

        for j in range(j_len):

            for i in range(0, i_len, self._ncol):
                tem_Qh = series.iloc[i:(i + self._ncol), j]
                tem_Qh = tem_Qh.reset_index(drop=True)

                Qh_tem.append(tem_Qh)

            tem2_Qh = pd.concat(Qh_tem, axis=1).T

            Qh_tem = []

            Qh_matrix.append(tem2_Qh)

        Qh_Locations = len(Qh_matrix)

        for L in range(Qh_Locations):
            Qh_matrix[L] = Qh_matrix[L].dropna()

        return Qh_matrix

    def corr(self, matrix):
        """
        method to create the correlation matrix of the historical inflows.
        The method attempts to repair non-positive-definite matrices when the Cholesky method is performed.

        Parameters:
        -----------
        matrix: narray
                matrix containing the historical inflows to calculate the correlation matrix.

        Returns:
        --------
        c_final: final correlation matrix of the historical inflows.
        """

        c = np.corrcoef(matrix, rowvar=False)
        eig = np.linalg.eigvals(c)

        trigger = None

        if np.min(eig) < 0:
            trigger = True

            # This method is used in https://github.com/julianneq/Kirsch-Nowak_Streamflow_Generator
            # k = np.min([np.min(np.real(eig)) -1 * np.finfo(float).eps])
            # c += c - k * np.eye(*c.shape)

            # At the end I am using this to do not have non-positive-definite matrices.
            # This solutions was given in https://github.com/iancze/Starfish/issues/26#issuecomment-479112856
            c += 1e-14 * np.eye(*c.shape)
            c_final = c

        if trigger is None:
            c_final = c

        return c_final

    def standardise_streamflows(self, shifted=None, new_dataframe=None):
        """
        method to standardise the historical inflows.

        Parameters:
        -----------
        new_dataframe: DataFrame, optional.
                       A pandas DataFrame containing the historical inflows for the different locations in the basing.
                       Each location must to be in a separately column in the DataFrame.
                       **Note**: be sure you remove the index column and any `str` column name.

        shifted: bool, default False
                If True, you are calling the `self.formatting_qh()` method internally.

        Returns:
        --------
        Log_Qh: Standardised and Log matrix of the historical inflows.
        """

        if new_dataframe is None:
            tem = self.Qh

        else:
            tem = new_dataframe

        self._Log_Qhmean = []
        self._Log_Qhstd = []

        Log_Qh = []
        mean_tem = []
        std_tem = []

        if shifted is False:
            Log_Qh = self.formatting_qh(shifted=False, new_dataframe=tem)

        if shifted is True:
            Log_Qh = self.formatting_qh(shifted=True, new_dataframe=tem)

        Qh_Locations = len(Log_Qh)

        for L in range(Qh_Locations):
            Log_tem = np.log(Log_Qh[L])

            mean_tem = Log_tem.mean()
            std_tem = Log_tem.std()

            Log_Qh[L] = (Log_tem - Log_tem.mean()) / Log_tem.std()

            self._Log_Qhmean.append(mean_tem)
            self._Log_Qhstd.append(std_tem)

        return Log_Qh

    def destandardise_streamflows(self, list_of_dataframes):
        """
        method to de-standardise the synthetic generated inflows.
        This method need be extended according to Kirsch et al. (2013) and/or Herman et al 2016

        Parameters:
        -----------
        list_of_dataframes: list of dataframes to de-standardise.
                       A list containing dataframes, each element in the list is a dataframe corresponding to a
                       location of streamflows.

        Returns:
        --------
        list_of_dataframes: List of dataframes with the Unwhitening synthetic generated inflows by location.
        """

        self._Log_Qhmean = []
        self._Log_Qhstd = []

        # At the moment this method take the _Log_Qhmean and _Log_Qhstd from the historical data
        Log_Qh = self.standardise_streamflows(shifted=False, new_dataframe=None)

        Qh_Locations = len(Log_Qh)

        for L in range(Qh_Locations):
            list_of_dataframes[L] = (list_of_dataframes[L] * self._Log_Qhstd[L]) + self._Log_Qhmean[L]
            list_of_dataframes[L] = np.exp(list_of_dataframes[L])

        return list_of_dataframes

    def to_dataframe(self, list_of_arrays):
        """
        method to create dataframes.

        Parameters:
        -----------
        list_of_arrays: list of arrays to convert in pd dataframes.

        Returns:
        --------
        df: Dataframe each column correspond to an inflows location.
        """

        Qh_Locations = len(list_of_arrays)

        for L in range(Qh_Locations):
            list_of_arrays[L] = list_of_arrays[L].T.ravel('F')
            list_of_arrays[L] = pd.DataFrame(list_of_arrays[L])

        df = pd.concat(list_of_arrays, axis=1)

        new_nme = []
        for i, col in enumerate(df.columns):
            if col == 0:
                new_nme.append(i)
                continue

            new_nme.append(col)

        df.columns = new_nme

        return df

    def random_matrix(self, shifted=None, new_dataframe=None):
        """
        random matrix to populate the intermediate matrix `M_(i,j)` in Kirsch et al. (2013).
        **Note**: This method create a list of random matrix by inflows location. However, latter only one matrix
                  is used to populate `M_(i,j)` and to maintain the cross-correlation among the locations.
                  #TODO modify the method to only return one matrix to not create confusions.

        Parameters:
        -----------
        new_dataframe: DataFrame, optional.
                       A pandas DataFrame containing the historical inflows for the different locations in the basing.
                       Each location must to be in a separately column in the DataFrame.
                       **Note**: be sure you remove the index column and any `str` column name.

        shifted: bool, default False
                If True, you are calling the `self.standardise_streamflows()` method internally.

        Returns:
        --------
        Random_matrix: at the moment a list of random matrix to populate `M_(i,j)`.
        """

        if new_dataframe is None:
            tem = self.Qh

        else:
            tem = new_dataframe

        Random_matrix = []

        if shifted is False:
            Log_Qh = self.standardise_streamflows(shifted=False, new_dataframe=tem)
            years = Log_Qh[0].shape[0]

        # TODO, this is not necessary. :confused:
        if shifted is True:
            Log_Qh = self.standardise_streamflows(shifted=True, new_dataframe=tem)
            years = Log_Qh[0].shape[0]  # + 1

        Qh_Locations = len(Log_Qh)

        rm_tem = np.zeros([years, self._ncol])

        for L in range(Qh_Locations):
            # tem = np.array(Log_Qh[L])

            for yr in range(years):
                rm_tem[yr, :] = np.random.randint(years, size=self._ncol)

            Random_matrix.append(rm_tem)

        return Random_matrix

    def generate_one_sampling(self, new_dataframe=None):
        """
        generate only one sampling of synthetic inflows by location.
        This method contain the whole methodology proposed in Kirsch et al. (2013).

        Parameters:
        -----------
        new_dataframe: DataFrame, optional.
                       A pandas DataFrame containing the historical inflows for the different locations in the basing.
                       Each location must to be in a separately column in the DataFrame.
                       **Note**: be sure you remove the index column and any `str` column name.

        shifted: bool, default False
                If True, you are calling the ... see what methods are you calling internally. :stuck_out_tongue_winking_eye:

        Returns:
        --------
        Qs: one final synthetic inflow sample by location.
        """

        if new_dataframe is None:
            tem = self.Qh

        else:
            tem = new_dataframe

        Log_Qs = []
        Log_Z = []
        Log_Z_shifted = []

        Random_matrix = self.random_matrix(shifted=False, new_dataframe=tem)
        Log_Qh = self.standardise_streamflows(shifted=False, new_dataframe=tem)
        Log_Qh_shifted = self.standardise_streamflows(shifted=True, new_dataframe=tem)

        years = Random_matrix[0].shape[0]

        Qh_Locations = len(Log_Qh)

        Qs_Uncorrelated = []
        Intermediate_matrix = []

        for L in range(Qh_Locations):
            tem_Ucor = np.zeros([years, self._ncol])
            Qs_Uncorrelated.append(tem_Ucor)

            tem_int = np.tile(Log_Qh[L], (years, 1))
            Intermediate_matrix.append(tem_int)

        for L in range(Qh_Locations):

            for yr in range(years):

                for i in range(self._ncol):
                    # Note I am using only one random matrix to populate the intermediate matrix
                    Qs_Uncorrelated[L][yr, i] = Intermediate_matrix[L][int(round(Random_matrix[0][yr, i])), i]

            Q = cholesky(self.corr(Log_Qh[L]), lower=False)

            tem = np.dot(Qs_Uncorrelated[L], Q)

            Log_Z.append(tem)

        Uncted_df = self.to_dataframe(Qs_Uncorrelated)

        Qs_Uncorrelated_shifted = self.formatting_qh(shifted=True, new_dataframe=Uncted_df)

        for L in range(Qh_Locations):
            Q_shifted = cholesky(self.corr(Log_Qh_shifted[L]), lower=False)

            tem_shifted = np.dot(Qs_Uncorrelated_shifted[L], Q_shifted)

            Log_Z_shifted.append(tem_shifted)

        Qs_tem = np.zeros([years - 1, self._ncol])

        for L in range(Qh_Locations):

            Log_Z[L] = pd.DataFrame(Log_Z[L])
            Log_Z[L] = Log_Z[L].iloc[1:, :].reset_index(drop=True)

            Log_Z_shifted[L] = pd.DataFrame(Log_Z_shifted[L])

            if self.monthly is True or self.weekly is True:
                div = int(self._ncol / 2)

                Qs_tem[:, :div] = Log_Z_shifted[L].loc[:, :div - 1]

                Qs_tem[:, div:] = Log_Z[L].loc[:, div:]

                Log_Qs.append(Qs_tem)

            if self.daily is True:
                div = int(np.floor(self._ncol / 2))

                Qs_tem[:, :div] = Log_Z_shifted[L].loc[:, :div - 1]

                Qs_tem[:, div:] = Log_Z[L].loc[:, div:]

                Log_Qs.append(Qs_tem)

        for L in range(Qh_Locations):
            Log_Qs[L] = pd.DataFrame(Log_Qs[L])

        Qs = self.destandardise_streamflows(Log_Qs)

        return Qs

    def generate_sample(self, new_dataframe=None, sample=None):
        """
        generate a sample of synthetic inflows by location.
        This method call internally `self.generate_one_sampling()` method.

        Parameters:
        -----------
        new_dataframe: DataFrame, optional.
                       A pandas DataFrame containing the historical inflows for the different locations in the basing.
                       Each location must to be in a separately column in the DataFrame.
                       **Note**: be sure you remove the index column and any `str` column name.

        sample: integer, default None
                sample size, this is used is a sample size was not initialise when was called the class.

        Returns:
        --------
        Qs_sample: Dataframe with the sample of synthetic inflows by location.
        """

        if sample is None:
            _sample = self.sample
        else:
            _sample = sample

        if new_dataframe is None:
            _new_dataframe = self.Qh
        else:
            _new_dataframe = new_dataframe

        List_Qs = []

        for i in range(_sample):

            tem = self.generate_one_sampling(_new_dataframe)

            for j in range(2):
                tem[j] = np.array(tem[j])

            tem2 = self.to_dataframe(tem)

            List_Qs.append(tem2)

        Qs_sample = pd.concat(List_Qs, axis=1)

        return Qs_sample
