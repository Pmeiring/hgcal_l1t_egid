import pandas as pd
import numpy as np
import xgboost as xg

class Boruta(object):
    '''
    Use the Boruta feature ranking and selection algorithm to systematically drop features
   
    attributes
    ----------
    :param X_train: DataFrame containing all X features required for initial training
    :param y_train: target columns (electron or PU flag)
    :param w_train: weights for training
    :param all_vars: list of all variables being considered for droppping
    :param i_ters: max number of times to run slimming algorithm. Useful if eventually the feature importance
                   for the every real variable is always higher than the shadow variable (avoid inf loop)
    :param n_trainings: number of times run classifier training and calculate importances
    :param max_vars_removed: max number of variables that may be removed
    :param running_vars: list of all variables that are to be kept. Updated as algo runs
    :param running_shadow_vars: list of current shadow variables. Updated as algo runs, based on running_vars
    :param tolerance: fraction of each column to be randomly shuffled when creating shadow variables
    '''

    def __init__(self, X_train, y_train, w_train, all_vars, i_iters=5, n_trainings=3, max_vars_removed=12, tolerance=0.1):

        self.X_train             = pd.DataFrame(X_train, columns=all_vars)
        self.y_train             = y_train
        self.w_train             = w_train
        self.all_vars            = all_vars
        self.i_iters             = i_iters
        self.n_trainings         = n_trainings
        self.max_vars_removed    = max_vars_removed
        self.running_vars        = all_vars
        self.running_shadow_vars = ['shadow_'+str(var) for var in all_vars] 
        self.tolerance           = tolerance

        #assert each element in all_vars is in X_train.columns

    def update_vars(self, varrs):
        '''
        Update current variables to train on, for both real and shadow sets
        '''
        self.running_vars = varrs
        self.running_shadow_vars = ['shadow_'+str(var) for var in varrs]

    def create_shadow(self):
        '''
        Take all X variables, creating copies and randomly shuffling them (or a subset of them)

        :return: dataframe 2x width and the names of the shadows for removing later
        '''

        real_running_x = self.X_train[self.running_vars]
        final_shuf_c  = []
        split_id = int(self.tolerance*real_running_x.shape[0])
        for c in real_running_x.columns:
            shuf_c, nominal_c = np.split(real_running_x[c], [split_id])
            np.random.shuffle(shuf_c)
            final_shuf_c.append(pd.concat([nominal_c, shuf_c]))

        #concat partially mixed columns into a single df
        x_shadow = pd.concat(final_shuf_c, axis=1)#, names = self.running_shadow_vars)
        x_shadow.columns = self.running_shadow_vars

        return pd.concat([real_running_x, x_shadow], axis=1)


    def run_trainings(self, train_params={}):
        '''
        Run classifier training for n_iters. Return the importances for each feature averaged over n_iters.
        We run this multiple times since the feature importance ranking is not deterministic.
        Note we need to set up the dict first with all real + shadow features still being considered,
        since the faetures importances for some features are zero and hence do not get returned; we then get
        keyErrors when trying to compare faetures later!
        '''

        #set up dict first!
        importances = {}
        for var in (self.running_vars+self.running_shadow_vars):
            importances[var] = []

        for n_iter in range(self.n_trainings):
            #create shadow set with remaining features
            x_mirror = self.create_shadow()
            print 'training classifier for iteration: {}'.format(n_iter)
            clf = xg.XGBClassifier(objective='binary:logistic', **train_params)
            print 'done'
            clf.fit(x_mirror, self.y_train, sample_weight=self.w_train)
            n_importance = clf.get_booster().get_score(importance_type='gain')

            #update importances
            for var in importances.keys():
                if var in n_importance.keys(): importances[var].append(n_importance[var])
                else: importances[var].append(0) #importance is zero if not returned from get_score()!

        print 'final importance df: {}'.format(pd.DataFrame.from_dict(importances))

        return pd.DataFrame.from_dict(importances)

    def check_stopping_criteria(self, removed_vars):
        '''
        check various stopping criteria e.g. how many features have been removed?, ...
        '''

        if len(removed_vars) >= self.max_vars_removed: return True
        else: return False

    def slim_features(self):
        '''
        Execute the slimming algorithm
        '''

        print 'starting Boruta algorithm'
        removed_vars     = []
        for i_iter in range(self.i_iters):
            print 'running slimming iteration {}'.format(i_iter)
            importances_df = self.run_trainings()

            #compare resulting feature importances
            new_running_vars = []
            for real_feature, shadow_feature in zip(self.running_vars, self.running_shadow_vars):
                print 'real {}: {}'.format(real_feature, np.mean(importances_df[real_feature]))
                print 'shadow {}: {}'.format(real_feature, np.mean(importances_df[shadow_feature]))
                if np.mean(importances_df[real_feature]) > np.mean(importances_df[shadow_feature]):
                    new_running_vars.append(real_feature)
                else:
                    print 'Removing feature: {} !'.format(real_feature)
                    removed_vars.append(real_feature)
            #update running vars for real and shadow
            self.update_vars(new_running_vars)
            if(self.check_stopping_criteria(removed_vars)): break
        print 'Boruta removed variables: {}'.format(removed_vars)

        #return optimal set of vars
        return self.running_vars 

    def __call__(self):
        self.slim_features()
