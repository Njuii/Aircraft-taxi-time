import random
import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Kolmogorov-Smirnov test
from scipy.stats import kstest

class PreProcess:
    def __init__(self, raw_df):

        # Normality Test
        self.processed = raw_df.copy()

        # Dropping duplicates and rows with no entries
        self.processed.drop_duplicates(inplace=True)
        self.processed.dropna(inplace=True)

        # Remove arrivals
        self.processed=self.processed.drop(self.processed[self.processed['type']=='arrival'].index)

        # Reset index
        self.processed.reset_index(inplace=True)
        self.processed.drop(labels='index',axis=1, inplace=True)

        # Day number extraction (Mon==0,...,Sun==6)
        date=slice(0,11)

        middle = np.asarray(self.processed.loc[:, "id"], dtype=str)
        insert = np.zeros(len(middle), dtype=float)
        for i in range(len(self.processed)):
            middle[i] = self.processed.loc[i, 'id'][date]
        df = pd.DataFrame()
        df.insert(0, "week_day", middle, True)
        df["week_day"] = pd.to_datetime(df["week_day"])
        insert = df["week_day"].dt.day_of_week

        # Assign weekday to airport activity of airport 
            # Friday busiest day
            # Saturday, Sunday and Monday next busiest
            # Thursday and Wednesday as mid
            # Tuesday as the quietest
            # (Data reflects this)
        # Using Data to by finding percentage of departures on day of week over the three month period
        day_counts=insert.value_counts()
        insert = np.asarray(insert, dtype=float)
        total_dep=len(self.processed)

        for i in range(len(self.processed)):
            for indent in range(6):
                if (insert[i] == float(indent)):
                    insert[i] = (day_counts[indent]/total_dep)*100
                    break
        self.processed.insert(len(self.processed.columns), "week_day", insert)

        # Assign runway hour to airport activity - same concept as weekday
        self.processed.insert(len(self.processed.columns), "hourly_congestion", np.asarray(self.processed.loc[:, "runway hour"], dtype=float))
        hour_counts = self.processed.loc[:, 'hourly_congestion'].value_counts()
        for i in range(len(self.processed)):
            for indent in range(23):
                if (self.processed.loc[i, "hourly_congestion"] == indent):
                    self.processed.loc[i ,"hourly_congestion"] = float((hour_counts[indent]/total_dep)*100)
                    break
        self.processed.drop(labels=['id', "type", "rwy"], axis=1, inplace=True)

        
        # Cleaning
        # Remove impossible data entries e.g. negative taxi time
        # Check for normality of different features:  
            # feature(s) that combine distances
            # feature(s) that combine the number of other aircraft
        # If normal clean using 5% level of significance (remove potential outliers)
        # If not normal clean using the IQR *1.5 (as standard)
        # PCA on combined dataset to see which features are the most relevant--> continue to tweak

        # Remove neagtive taxi time
        for i in range(len(self.processed)):
            if self.processed['taxi_time'][i]<0:
                self.processed.drop(i,axis='index',inplace=True)

        # reindex
        self.processed.reset_index(inplace=True)
        self.processed.drop(labels='index',axis=1, inplace=True)

        # Isolate the last columns, for convenience i've correlated
        # each string (column) to an index in an array plus 2 more for
        # the none and many special cases
        #db_moving = self.processed.loc[:, "other_moving_ac":"NArrArr"]
        #indices_max = self.processed.loc[:, "QDepDep":"NArrArr"].to_numpy().argmax(axis=1) 
        #dom = indices_max + 2
        #names = np.append(["None", "Many"], self.processed.loc[:, "QDepDep":"NArrArr"].columns)
        #
        ## Need to find when there are no other moving aircraft which
        ## fit into the 8 original features.
        #none = np.zeros(self.processed.shape[1], dtype=bool)
        #none = (self.processed.loc[:, "QDepDep":"NArrArr"].to_numpy().sum(axis=1) == 0)
        #dom[none] = 0
        #
        ## Need to find when there are multiple categories which are
        ## dominant. Bitwise condition is checking to make sure non zero
        ## and the number of occurences is greater than 1
        #multiple = np.zeros_like(none)
        #max_values = self.processed.loc[:, "QDepDep":"NArrArr"].max(axis=1)
        #multiple = ((max_values > 0) & (self.processed.loc[:, "QDepDep":"NArrArr"].eq(max_values, axis=0).sum(axis=1) > 1)) 
        #dom[multiple] = 1
        #
        #ins = names[dom]
        #self.processed.insert(len(self.processed.columns), "DomVal", dom, True)

        # List of features to check for normal distribution
        self.to_check=['distance','shortest path','distance_gate','distance_else',
                  'other_moving_ac','QDepDep','QDepArr','QArrDep','QArrArr',
                  'NDepDep','NDepArr','NArrDep','NArrArr','taxi_time']

        flag = False
        # column_headers = list(test_feats.columns.values)
        for i in self.to_check:
            if not (self.check_normality(self.processed[i])):
                flag = True
                print(f"{i}is not normal")

        if not flag:
            print("All features are normal")
        print("\n")

        # Clean the dist and move features
        # All of the 4 features are normal (based on K-S test)
        for i in self.to_check:
            if self.check_normality(self.processed[i]):
                self.clean_outliers_z_score(self.processed[i])
                self.reindex(self.processed)
            else:
                test_feats = self.clean_outliers_iqr(self.processed[i])
                self.reindex(self.processed)

    # Check normality
    def check_normality(self, feature, alpha=0.05):
        p = kstest(feature, cdf='norm')[1]
        #print(feature, p)
        return p < alpha

    # Clean outliers with 1.5 * IQR
    def clean_outliers_iqr(self, feature):
        q1 = feature.quantile(0.25)
        q3 = feature.quantile(0.75)
        iqr_val = q3 - q1
        lower_bound = q1 - 1.5 * iqr_val
        upper_bound = q3 + 1.5 * iqr_val
        for i in range(len(feature)):
            if feature[i] <= lower_bound or feature[i] >= upper_bound:
                self.processed.drop(i,axis='index',inplace=True)

    # Clean outliers with z-score
    def clean_outliers_z_score(self, feature):
        mu = np.mean(feature)
        sd = np.std(feature)
        for i in range(len(feature)):
            if np.abs((feature[i] - mu)) > 3 *sd:
                self.processed.drop(i,axis='index',inplace=True)

    # Reindex
    def reindex(self, feature_df):
        feature_df.reset_index(inplace=True)
        feature_df.drop(labels='index',axis=1, inplace=True)

class Analysis:
    def __init__(self, csv):
        self.active_features = []

        self.raw_df = pd.read_csv(csv)
        self.pre_proc_df = PreProcess(self.raw_df).processed

                # Removing unnecessary features
        # id
        # Type
        # Gate --> not high enough precision for difference between runway hour
        # rwy_day-->  encompassed in day of week (too dependent on month) (also contained errors)
        # angle--> not consistent and doesnt convey info
        # Angle error--> single value dominated 
        # (distances-pending)--> identify bottleneck and remove others
        # rwy --> str of rwy_num (same data)
        # operation mode--> encompassed in type (since this is for departures remove) single value dominated
        # rwy_num --> Dominated by a single value, doesn't add enough variation or capture any useable behaviour
        # QArr + NArr --> Arrivals have been dropeed, these values are all zeros, they can be dropped

        # Keep taxi time for now to check for any outliers to drop
        self.pre_proc_df.drop(labels=['runway hour','gate (block) hour','rwy_day','angle','angle error',
                                      'operation_mode','rwy_num','QArrDep','QArrArr','NArrDep','NArrArr'],axis=1,inplace=True)
        
        self.saved_df = self.pre_proc_df
        self.active_df = self.pre_proc_df
        self.principal_Df = self.pre_proc_df

    def PCA_feature_probe(self, feature_name, database, fig_axes):
        # Extract values of features in column
        # Assign colour to feature on a sliding scale (e.g. blue=smallest val  <---> red=largest val)
            # feature--> pd.Series(all values of features)
            # target--> Series of unique values of features (sorted) numpy array
            # Colours--> corresponding colours
        feature = database[feature_name]
        targets = feature.unique()
        targets = np.sort(targets)
        colours = plt.cm.rainbow(np.linspace(0, 1, len(targets)))

        # Dictionary {key=feature : value = [index1, index2,..., indexn]}
        # Put indices corresponding to each unque value in a feature column in to a dictionary
        dct = {}
        for index, value in enumerate(feature):
            if value in dct:
                dct[value].append(index)
            else:
                dct[value] = [index]


        # For each unique feature value, plot corresponding PC1 against PC2 for all data points
        # Use colour assgned previously to visualise magnitude of the original feature (peeking into its original value)
        for target, colour in zip(targets,colours):
            fig_axes[1].scatter(self.principal_Df.loc[dct[target], 'principal component 1'], 
                                self.principal_Df.loc[dct[target], 'principal component 2'], color = colour, s = 3)


        # Plot settings
            # checks for strings in the values of any of the features (useful when considering 'operation_mode' feature)
                # why targets.size < 3 tho? 
                # - I was just hard coding the boolean string type in
            # If string present, data is normalized into either 1 colour or second colour
            # linearly normalizes colours on sliding scale corresponding to max and min of feature value
            # adds a colour bar to explain what colours mean
        fig_axes[1].set_title(feature_name)
        contains_strings = False
        if targets.size < 3:
            contains_strings = any(isinstance(item, str) for item in targets)
        if contains_strings:
            norm = mpl.colors.Normalize(vmin=np.min(0), vmax=np.max(1))
        else:
            norm = mpl.colors.Normalize(vmin=np.min(targets), vmax=np.max(targets))
    
        cmap = plt.cm.rainbow
        fig_axes[0].colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=fig_axes[1], orientation='horizontal')

    def PCA_OnFeatures(self, acceptable=0.85, output=False, detailed=False):
        # 5th iteration of distance features, 3 PCs with [shortest path,distance_gate,distance_long,distance_else]
        # Normalizing the distance features
        feats = self.active_features
        features=np.array(feats)

        x = self.active_df.loc[:, features].values
        x = StandardScaler().fit_transform(x) # normalising the features

        feat_cols = feats
        normalised_dist_db = pd.DataFrame(x,columns=feat_cols)

        # PCA
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        self.principal_Df = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

        if output:
            if (round(sum(pca.explained_variance_ratio_),3) > acceptable):
                print("Results for: " + " ".join(self.active_features))
                print('Explained variation per principal component:{}'.format(pca.explained_variance_ratio_))
                print(f'Total Variance captured {round(sum(pca.explained_variance_ratio_)*100,3)}%')
                if detailed:
                    # Conclusion (Fifth iteration):
                    print('')
                    print(f"Mean = {np.mean(x)} Standard Deviation = {np.std(x)}")
                    print('')

                    print('Matrix showing how much each feature is correlated to the principle components:')
                    print(pca.components_)
                    print('')
                    print('Correlation matrix of features in PCA:')
                    print(np.corrcoef([normalised_dist_db[i] for i in feats]))
                    print('')
                    print(f"Eigenvalues of each principle component: {pca.explained_variance_}")
                    print('')
                    print(f"Measure of noise covarience in current iteration leftover data: {pca.noise_variance_}")
                    print('')
                    print('-----------------------------------------------------------------------------')
        return (round(sum(pca.explained_variance_ratio_),3))

    def IteratePCA_Combinations(self, name, features, acceptable=0.85, min_feats=3, graph=False):
        largest_set = []
        largest_variance = 0
        for L in range(len(features) + 1):
            if L > min_feats - 1:
                for subset in itertools.combinations(features, L):
                    self.active_features = subset
                    var = self.PCA_OnFeatures(acceptable, output=True, detailed=True) 
                    if (var > largest_variance):
                        largest_variance = var
                        largest_set = np.array(subset)
        self.active_features = largest_set
        self.PCA_OnFeatures(acceptable)
        self.UpdateActiveWithPCA(name, save=True)
        print(" ".join(self.active_features) + " Used for PCA of set: \n " + " ".join(features))
        print("With a total Variance of: " + str(largest_variance*100) + "%\nThis reduces the features by " + str(-2+len(largest_set)))
        print("\n")

        if graph:
            self.PlotPCA_Group()

        return largest_set
            
    def PlotPCA_Group(self):
        # X:PC1 , Y:PC2
        n = int(np.ceil(np.sqrt(len(self.active_features))))

        fig, ax = plt.subplots(n, n, figsize=(15,15), constrained_layout=True)
        # Calls PCA_Feature_Probe function and iterates through all the features that PCA was performed on (e.g. features related to distance)
        for index in range(n*n):
            if index < len(self.active_features):
                fig_ax = (fig, ax[int(np.floor(index / n)), index % n])
                print("Working..." + self.active_features[index])
                self.PCA_feature_probe(self.active_features[index], self.active_df, fig_ax)
                print("Finished..." + self.active_features[index])

        print("Working...subplots")
        fig.suptitle("Principal Component Analysis",fontsize=20)
        fig.supxlabel('Principal Component - 1',fontsize=20)
        fig.supylabel('Principal Component - 2',fontsize=20)
        print("Finished...subplots")

    def UpdateActiveWithPCA(self, name, save=False):

        self.active_df = pd.concat([self.active_df,self.principal_Df], axis=1)
        self.active_df.drop(labels=self.active_features, axis=1, inplace=True)
        mapping={
            'principal component 1' : name + ' PC1',
            'principal component 2' : name + ' PC2',
        }
        self.active_df.rename(mapper=mapping, axis=1, inplace=True)
        if save: self.saved_df = self.active_df

    def ReduceProcessedSize(self, graph=False):
        random.seed(69420)
        grouped = np.array([(a, b) for a, b in zip(self.saved_df.loc[:, "taxi_time"], self.saved_df.index.values)])
        sorted = np.argsort(grouped[:, 0])
        sorted = grouped[sorted]
        rnd = random.sample(range(0, len(sorted)), 1000)
        final = sorted[rnd, 1]
        self.saved_df = self.saved_df.loc[final, :]
        # self.saved_df.to_csv("./final.csv")

        if graph:
            plt.hist(self.saved_df.loc[:, "taxi_time"], bins=250)
            plt.hist(self.saved_df.loc[final, "taxi_time"], bins=250)
            plt.xlabel("Taxi time")
            plt.ylabel("Frequency")
            plt.title("taxi time sorted distribution")
            plt.legend(["Full", "Reduced"])
            plt.show();




def main():
    an = Analysis("features_full.csv")
    
    distance = ['distance', 'distance_long','shortest path','distance_gate','distance_else']
    congestion = ['other_moving_ac', 'QDepDep', 'QDepArr', 'NDepDep', 'NDepArr']

    an.IteratePCA_Combinations("distance", distance, min_feats=4, acceptable=0.5)
    an.IteratePCA_Combinations("congest", congestion, min_feats=4, acceptable=0.5)
    an.ReduceProcessedSize()


    # normalize final df (didn't define fancy functions for this)
    features_fin = [col for col in an.saved_df]
    features_fin.pop(0)

    y = an.saved_df.reset_index().copy()
    y.drop(labels='index',axis=1, inplace=True)
    x = an.saved_df.loc[:, features_fin].values.copy()
    x = StandardScaler().fit_transform(x)

    normalised_df= pd.DataFrame(x,columns=features_fin)
    normalised_df
    normalised_df['taxi_time'] = y['taxi_time']

    print(normalised_df)
    normalised_df.to_csv("./final.csv")


    # print(an.saved_df)

if __name__ == "__main__":
    main()