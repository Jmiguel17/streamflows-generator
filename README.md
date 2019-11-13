# Streamflows Generator
Python implementation of a synthetic streamflow generator based on Cholesky decomposition.
Method developed by Kirsch et al. (2013)

At the moment the implementation is not so efficient :disappointed: could be many things to correct. 
However, it can generate 10000s of samples for different locations in a couple of seconds.    

This class is inspired by the Matlab implementation found in:
* https://github.com/jdherman/Qsynth, and
* https://github.com/julianneq/Kirsch-Nowak_Streamflow_Generator

Parameters:
-----------
**dataframe**: DataFrame
                A pandas DataFrame containing the historical inflows for the different locations in the basing.
                Each location must to be in a separately column in the DataFrame.
                **Note**: be sure you remove the index column and any `str` column name.

**monthly**: bool, default False
            If True your historical inflows are in a monthly timestep.

**weekly**: bool, default False
            If True your historical inflows are in a weekly timestep.

**daily**: bool, default False
            If True your historical inflows are in a daily timestep.

**sample_size**: integer, default None
                Number of sample to generate by each inflows locations.

**Citations**:
--------------
* Kirsch, B. R., G. W. Characklis, and H. B. Zeff (2013), Evaluating the impact of alternative hydro-climate scenarios on transfer agreements:
  Practical improvement for generating synthetic streamflows, Journal of Water Resources Planning and Management, 139(4), 396–406.

* Herman, J.D, H.B. Zeff, J.R. Lamontagne, P.M. Reed, and G.W. Characklis, Synthetic drought scenario generation to support bottom-up
  water supply vulnerability assessments, Journal of Water Resources Planning and Management, 142(11), 04016050, 2016.
  
