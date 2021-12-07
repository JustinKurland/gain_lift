import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas_flavor as pf

@pf.register_dataframe_method
def print_decile_labels():

    print(
        "Column Label Information:\n\n",
        "cnt_events     : Count of events in a particular decile\n",
        "cnt_true       : Count of true labels in a particular decile\n",
        "cnt_false      : Count of false labels in a particular decile\n",
        "cum_apps       : Cumulative sum of events decile-wise \n",
        "cum_true       : Cumulative sum of true labels decile-wise \n",
        "cum_false      : Cumulative sum of false decile-wise \n",
        "cum_pct_events : Cumulative sum of percentages of events decile-wise \n",
        "true_rate      : Rate of true labels in a particular decile [(cnt_true/cnt_events)*100]\n",
        "gain           : Gain, cumulative sum of percentages of true/false labels decile-wise \n",
        "loss           : Loss, cumulative sum of percentages of false/false labels decile-wise \n",
        "lift           : Lift, cumuative lift value decile-wise"
         )

@pf.register_dataframe_method
def print_quartile_labels():

    print(
        "Column Label Information:\n\n",
        "cnt_events     : Count of events in a particular quartile\n",
        "cnt_true       : Count of true labels in a particular quartile\n",
        "cnt_false      : Count of false labels in a particular quartile\n",
        "cum_apps       : Cumulative sum of events quartile-wise \n",
        "cum_true       : Cumulative sum of true labels quartile-wise \n",
        "cum_false      : Cumulative sum of false labels quartile-wise \n",
        "cum_pct_events : Cumulative sum of percentages of events quartile-wise \n",
        "true_rate      : Rate of true labels in a particular quartile [(cnt_true/cnt_events)*100]\n",
        "gain           : Gain, cumulative sum of percentages of true labels quartile-wise \n",
        "loss           : Loss, cumulative sum of percentages of false labels quartile-wise \n",
        "lift           : Lift, cumuative lift value quartile-wise"
         )

@pf.register_dataframe_method
def gain_lift_table(y_true, 
                    y_prob,
                    category      = 0, 
                    output_type   = "deciles",
                    labels        = True, 
                    round_decimal = 3):
    """
    Generates the Gain and Lift Table from labels and probabilities for either
    deciles or quartiles by sorting events by their predicted 
    probabilities, in decreasing order from highest (closest to one) to 
    lowest (closest to zero). Splitting the events into equally sized bins, 
    we create groups containing the same numbers of events, for example, 10 decile 
    groups each containing 10% of the applicant base or 4 groups each containing 25%
    of the events.
    
    Args:
        y_true ([numpy.ndarray]):
            Ground truth (correct) target values.
            
        y_prob ([numpy.ndarray]):
            Predicted probabilities for each class returned by a classifier.
        
        category ([int, optional]):
            The category of interest. Defaults to the 0 class/category.
            
        labels ([bool, optional]): 
            If True, prints a legend for the abbreviations of decile or 
            quantile table column names. Defaults to True.
            
        round_decimal (int, optional): 
            The decimal precision to which the result is needed. Defaults 
            to '3'.
            
    Returns:
        if output_type=="decile":
        [pandas.DataFrame] The dataframe of deciles and related information.
 
        if output_type=="quartile":
        [pandas.DataFrame] The dataframe of quartiles and related information.

    Example:
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn import tree
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=3)
        >>> clf = tree.DecisionTreeClassifier(max_depth=1,random_state=3)
        >>> clf = clf.fit(X_train, y_train)
        >>> y_prob = clf.predict_proba(X_test)
        >>> gain_lift_table(y_test, y_prob[:,0])
    """
    
    # Outputs Decile Table
    if output_type=="deciles":

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        df           = pd.DataFrame()
        df['y_true'] = y_true
        df['y_prob'] = y_prob

        df.sort_values('y_prob', ascending=False, inplace=True)
        df['decile'] = np.linspace(1, 11, len(df), False, dtype=int)

        # dt abbreviation for gain_lift_table
        dt = df.groupby('decile').apply(lambda x: pd.Series([
            np.size(x['y_prob']),
            np.sum(x['y_true']),
            np.size(x['y_true'][x['y_true'] == category]),
            np.min(x['y_prob']),
            np.max(x['y_prob']),
            np.mean(x['y_prob'])
        ],
            index=(["cnt_events",
                    "cnt_false",
                    "cnt_true",
                    "prob_min", 
                    "prob_max", 
                    "prob_avg"
                    ])
        )).reset_index()

        tmp           = df[['y_true']].sort_values('y_true', ascending=False)
        tmp['decile'] = np.linspace(1, 11, len(tmp), False, dtype=int)

        dt['true_rate']      = round(dt['cnt_true'] * 100 / dt['cnt_events'], round_decimal)
        dt['misrep_rate']    = round(dt['cnt_false'] * 100 / dt['cnt_events'], round_decimal)
        dt['cum_apps']       = np.cumsum(dt['cnt_events'])
        dt['cum_true']       = np.cumsum(dt['cnt_true'])
        dt['cum_false']      = np.cumsum(dt['cnt_false'])
        dt['cum_pct_events'] = round(dt['cum_apps'] * 100 / np.sum(dt['cnt_events']), round_decimal)
        dt['gain']           = round(dt['cum_true'] * 100 / np.sum(dt['cnt_true']), round_decimal)
        dt['loss']           = round(dt['cum_false'] * 100 / np.sum(dt['cnt_false']), round_decimal)
        dt['lift']           = round(dt['gain'] / dt['cum_pct_events'], round_decimal)

        # Arrange Column Order
        dt = dt[['decile','cnt_events', 'cnt_true', 'cnt_false', 'cum_apps', 'cum_true', 'cum_false', 'cum_pct_events', 'true_rate', 'misrep_rate', 'gain', 'loss',  'lift']]

        if labels is True:
            print_decile_labels()

        return dt

    # Outputs Quartile Table
    if output_type=="quartiles":
        
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        df = pd.DataFrame()
        df['y_true'] = y_true
        df['y_prob'] = y_prob

        df.sort_values('y_prob', ascending=False, inplace=True)
        df['quartile'] = np.linspace(1, 5, len(df), False, dtype=int)

        # qt abbreviation for quartile_table
        qt = df.groupby('quartile').apply(lambda x: pd.Series([
            np.size(x['y_prob']),
            np.sum(x['y_true']),
            np.size(x['y_true'][x['y_true'] == category]),
            np.min(x['y_prob']),
            np.max(x['y_prob']),
            np.mean(x['y_prob'])
        ],
            index=(["cnt_events",
                    "cnt_false",
                    "cnt_true",
                    "prob_min", 
                    "prob_max", 
                    "prob_avg"
                    ])
        )).reset_index()

        tmp             = df[['y_true']].sort_values('y_true', ascending=False)
        tmp['quartile'] = np.linspace(1, 5, len(tmp), False, dtype=int)

        qt['true_rate']      = round(qt['cnt_true'] * 100 / qt['cnt_events'], round_decimal)
        qt['misrep_rate']    = round(qt['cnt_false'] * 100 / qt['cnt_events'], round_decimal)
        qt['cum_apps']       = np.cumsum(qt['cnt_events'])
        qt['cum_true']       = np.cumsum(qt['cnt_true'])
        qt['cum_false']      = np.cumsum(qt['cnt_false'])
        qt['cum_pct_events'] = round(qt['cum_apps'] * 100 / np.sum(qt['cnt_events']), round_decimal)
        qt['gain']           = round(qt['cum_true'] * 100 / np.sum(qt['cnt_true']), round_decimal)
        qt['loss']           = round(qt['cum_false'] * 100 / np.sum(qt['cnt_false']), round_decimal)
        qt['lift']           = round(qt['gain'] / qt['cum_pct_events'], round_decimal)
        qt['quartile_pct']   = [0.25, 0.50, 0.75, 1.00]

        # Change Column Order
        qt = qt[['quartile','quartile_pct','cnt_events', 'cnt_true', 'cnt_false', 'cum_apps', 'cum_true', 'cum_false', 'cum_pct_events', 'true_rate', 'misrep_rate', 'gain', 'loss',  'lift']]

        if labels is True:
            print_quartile_labels()

        return qt


@pf.register_dataframe_method
def plot_lift(y_true, 
              y_prob,
              category         = 0, 
              output_type      = "deciles",
              line_color       = '#0E4978',
              line_width       = 1,
              title            = 'Lift Chart', 
              title_fontsize   = 14, 
              title_weight     = "bold",
              text_fontsize    = 10,
              x_ax_title       = "Events",
              x_ax_weight      = "bold",
              y_ax_title       = "Lift",
              y_ax_weight      = "bold",
              ax_title_size    = 12,
              grid_color       = "#EEEEEE",
              grid_line_width  = 0.8,
              legend_size      = 'small',
              figsize          = None):
    """
    Generates a cumulative lift plot from labels and probabilities for either
    deciles or quartiles. It measures how much better one can expect to do with 
    the predictive model compared to without a model. It is the ratio of gain 
    percentage to the random expectation percentage at a given decile/quantile 
    level. The random expectation is that the xth decile/quantile is x%.
    
    Args:
        y_true ([numpy.ndarray]):
            Ground truth (correct) target values.
            
        y_prob ([numpy.ndarray]):
            Predicted probabilities for each class returned by a classifier.
            
        category ([int, optional]):
            The category of interest. Defaults to the 0 class/category.
        
        output_type ([str, optional]): 
            String of either 'deciles' or 'quantiles' specifying if decile values or quantile  
            values are plotted. Default is 'deciles'
    
        line_color ([str, optional)]: 
            String specifying the color of the lines that will be plotted. Defaults to black.
      
        line_width ([float, optional)]: 
            Width of the lines that will divide each cell. Defaults to 1.

        title ([str, optional]): 
            Title of the generated plot. Defaults to "Lift Chart".
            
        title_fontsize ([str or int, optional]): 
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 14.
            
        text_fontsize ([string or int, optional]): 
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 10.
            
        x_ax_title ([str, optional]): 
            X-axis title of the generated plot. Defaults to "events".      
        
        x_ax_weight ([str, optional]): 
            String specifying the x-axis title weight if title is desired, choices include the default, which is 'bold', 
            'light', 'normal', 'medium', 'semibold', 'heavy', and 'black'. 
     
        y_ax_title ([str, optional]): 
            Y-axis title of the generated plot. Defaults to "Lift".    
        
        y_ax_weight ([str, optional]): 
            String specifying the y-axis title weight if title is desired, choices include the default, which is 'bold', 
            'light', 'normal', 'medium', 'semibold', 'heavy', and 'black'.
     
        ax_title_size ([int, optional]): 
            Integer for axis title font size. Default value is 12.
   
        grid_color ([str, optional]):
            String specifying the color of the lower value of the color scale. Defaults to '#EEEEEE'.

        grid_line_width ([float, optional)]: 
            Width of the grid lines. Defaults to 0.8.
            
        legend_size ([str or int, optional]):        
            String {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'} or integer for the legend size. Defaults to 'small'.

        figsize ([int,int]): 
            Tuple denoting figure size of the plot, e.g. (6, 6). Defaults to `None`.
            
    Returns:
        if output_type=="decile":
        [matplotlib.axes._subplots.AxesSubplot] Plot of cumulative lift at each decile.
        
        if output_type=="quartile":
        [matplotlib.axes._subplots.AxesSubplot] Plot of cumulative lift at each quartile.

    Example:
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn import tree
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=3)
        >>> clf = tree.DecisionTreeClassifier(max_depth=1,random_state=3)
        >>> clf = clf.fit(X_train, y_train)
        >>> y_prob = clf.predict_proba(X_test)
        >>> plot_lift(y_test, y_prob[:,0])
    """
    # Decile Lift Plot
    if output_type=="deciles":
    
        pl = gain_lift_table(y_true,y_prob,labels=False, category=category, output_type="deciles")
        plt.plot(pl.decile.values, pl.lift.values, marker='o', label='Model', color=str(line_color), linewidth=line_width)
        plt.plot([0, 10], [1, 1], 'k--', marker='o', label='Baseline')
        plt.title(title, fontsize=title_fontsize, fontweight=title_weight)
        plt.xticks([0,2,4,6,8,10],['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.xlabel(x_ax_title, fontsize=text_fontsize, fontweight=x_ax_weight)
        plt.ylabel(y_ax_title, fontsize=text_fontsize, fontweight=y_ax_weight)
        plt.legend(loc=0, fontsize=legend_size)
        plt.grid(which='major', color=grid_color, linewidth=grid_line_width)
        
    # Quartile Lift Plot
    if output_type=="quartiles":
    
        pl = gain_lift_table(y_true,y_prob,labels=False, category=category, output_type="quartiles")
        plt.plot(pl.quartile.values, pl.lift.values, marker='o', label='Model', color=str(line_color), linewidth=line_width)
        plt.plot([0, 4], [1, 1], 'k--', marker='o', label='Baseline')
        plt.title(title, fontsize=title_fontsize, fontweight=title_weight)
        plt.xticks([0,1,2,3,4],['0%', '25%', '50%', '75%', '100%'])
        plt.xlabel(x_ax_title, fontsize=text_fontsize, fontweight=x_ax_weight)
        plt.ylabel(y_ax_title, fontsize=text_fontsize, fontweight=y_ax_weight)
        plt.legend(loc=0, fontsize=legend_size)
        plt.grid(which='major', color=grid_color, linewidth=grid_line_width)

@pf.register_dataframe_method
def plot_gain(y_true, 
              y_prob,
              category         = 0,
              output_type      = "deciles",
              line_color       = '#0E4978',
              line_width       = 1,
              title            = 'Gain Chart',
              title_fontsize   = 14,
              title_weight     = "bold",
              text_fontsize    = 10,
              x_ax_title       = "Events",
              x_ax_weight      = "bold",
              y_ax_title       = "Gain",
              y_ax_weight      = "bold",
              ax_title_size    = 12,
              x_label          = 0.5,
              y_label          = 1.48,
              grid_color       = "#EEEEEE",
              grid_line_width  = 0.8,
              legend_size      = 'small',
              figsize          = None):
    """
    Generates the cumulative gain plot from labels and probabilities for either
    deciles or quartiles. Gain at a given decile/quantile level is the ratio of 
    the cumulative number of targets (events) up to that decile/quantile to the 
    total number of targets (events) in the entire data set.
    
    Args:    
        y_true ([numpy.ndarray]):
            Ground truth (correct) target values.
            
        y_prob ([numpy.ndarray]):
            Predicted probabilities for each class returned by a classifier.
            
        category ([int, optional]):
            The category of interest. Defaults to the 0 class/category.
        
        output_type ([str, optional]): 
            String of either 'deciles' or 'quantiles' specifying if decile values or quantile  
            values are plotted. Default is 'deciles'
    
        line_color ([str, optional)]: 
            String specifying the color of the lines that will be plotted. Defaults to black.
      
        line_width ([float, optional)]: 
            Width of the lines that will divide each cell. Defaults to 1.

        title ([str, optional]): 
            Title of the generated plot. Defaults to "Lift Chart".
            
        title_fontsize ([str or int, optional]): 
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 14.
            
        text_fontsize ([string or int, optional]): 
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 10.
            
        x_ax_title ([str, optional]): 
            X-axis title of the generated plot. Defaults to "events".      
        
        x_ax_weight ([str, optional]): 
            String specifying the x-axis title weight if title is desired, choices include the default, which is 'bold', 
            'light', 'normal', 'medium', 'semibold', 'heavy', and 'black'. 
     
        y_ax_title ([str, optional]): 
            Y-axis title of the generated plot. Defaults to "Gain".    
        
        y_ax_weight ([str, optional]): 
            String specifying the y-axis title weight if title is desired, choices include the default, which is 'bold', 
            'light', 'normal', 'medium', 'semibold', 'heavy', and 'black'.
     
        ax_title_size ([int, optional]): 
            Integer for axis title font size. Default value is 12.
   
        grid_color ([str, optional]):
            String specifying the color of the lower value of the color scale. Defaults to '#EEEEEE'.

        grid_line_width ([float, optional)]: 
            Width of the grid lines. Defaults to 0.8.
            
        legend_size ([str or int, optional]):        
            String {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'} or integer for the legend size. Defaults to 'small'.

        figsize ([int,int]): 
            Tuple denoting figure size of the plot, e.g. (6, 6). Defaults to `None`.
            
    Returns:
        if output_type=="decile":
        [matplotlib.axes._subplots.AxesSubplot] Plot of cumulative gain at each decile.
        
        if output_type=="quartile":
        [matplotlib.axes._subplots.AxesSubplot] Plot of cumulative gain at each quartile.
    
    Example:
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn import tree
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=3)
        >>> clf = tree.DecisionTreeClassifier(max_depth=1,random_state=3)
        >>> clf = clf.fit(X_train, y_train)
        >>> y_prob = clf.predict_proba(X_test)
        >>> plot_gain(y_test, y_prob[:,0])
    """

    # Decile Gain Plot
    if output_type=="deciles":
    
        pcg = gain_lift_table(y_true,y_prob,labels=False, category=category, output_type="deciles")
        plt.plot(np.append(0, pcg.decile.values), np.append(0, pcg.gain.values), marker='o', label='Model', color=str(line_color), linewidth=line_width)
        plt.plot([0, 10], [0, 100], 'k--', marker='o', label='Baseline')
        for i, txt in enumerate(pcg['cum_true'].values.astype(np.int64).tolist()):
                plt.annotate(txt, (pcg['decile'].index[i]+.5, pcg['cum_true'].values[i]*y_label),fontsize=9)
        plt.title(title, fontsize=title_fontsize, fontweight=title_weight)
        plt.xticks([0,2,4,6,8,10],['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.xlabel(x_ax_title, fontsize=text_fontsize, fontweight=x_ax_weight)
        plt.gca().set_yticklabels(['{:.0f}%'.format(x*1) for x in plt.gca().get_yticks()]) 
        plt.ylabel(y_ax_title, fontsize=text_fontsize, fontweight=y_ax_weight)
        plt.legend(loc=0, fontsize=legend_size)
        plt.grid(which='major', color=grid_color, linewidth=grid_line_width)
        
    # Quartile Gain Plot
    if output_type=="quartiles":
    
        pcg = gain_lift_table(y_true,y_prob,labels=False, category=category, output_type="quartiles")
        plt.plot(np.append(0, pcg.quartile.values), np.append(0, pcg.gain.values), marker='o', label='Model', color=str(line_color), linewidth=line_width)
        plt.plot([0, 4], [0, 100], 'k--', marker='o', label='Baseline')
        for i, txt in enumerate(pcg['cum_true'].values.astype(np.int64).tolist()):
                plt.annotate(txt, (pcg['quartile'].index[i]+x_label, pcg['cum_true'].values[i]*y_label),fontsize=9)
        plt.title(title, fontsize=title_fontsize, fontweight=title_weight)
        plt.xticks([0,1,2,3,4],['0%', '25%', '50%', '75%', '100%'])
        plt.xlabel(x_ax_title, fontsize=text_fontsize, fontweight=x_ax_weight)
        plt.gca().set_yticklabels(['{:.0f}%'.format(x*1) for x in plt.gca().get_yticks()]) 
        plt.ylabel(y_ax_title, fontsize=text_fontsize, fontweight=y_ax_weight)
        plt.legend(loc=0, fontsize=legend_size)
        plt.grid(which='major', color=grid_color, linewidth=grid_line_width)
    
@pf.register_dataframe_method
def gain_lift_report(y_true, 
                     y_prob,
                     category         = 0,
                     output_type      = "deciles",
                     line_color       = '#0E4978',
                     line_width       = 1,
                     title_fontsize   = 14,
                     title_weight     = "bold",
                     text_fontsize    = 10,
                     x_ax_title       = "Events",
                     x_ax_weight      = "bold",
                     y_lift_ax_title  = "Lift",
                     y_gain_ax_title  = "Gain",
                     y_ax_weight      = "bold",
                     y_lab_pad_right  = -525,
                     ax_title_size    = 12,
                     grid_color       = "#EEEEEE",
                     grid_line_width  = 0.8,
                     labels           = True, 
                     plot_style       = None, 
                     round_decimal    = 3,
                     legend_size      = 'small',
                     figsize          = (16, 10)):
    """
    Generates gain/lift table and 2 plots (Gain & Lift) from labels and probabilities.
    The lift measures how much better one can expect to do with the predictive model 
    compared to without a model. It is the ratio of gain percentage to the random expectation 
    percentage at a given decile/quantile level. The random expectation is that the xth 
    decile/quantile is x%. While the gain provides a ratio of the cumulative number of targets 
    (events) up to that decile/quantile to the total number of targets (events) in the entire 
    data set.
    
    Args:
        y_true ([numpy.ndarray]):
            Ground truth (correct) target values.
            
        y_prob ([numpy.ndarray]):
            Predicted probabilities for each class returned by a classifier.
            
        category ([int, optional]):
            The category of interest. Defaults to the 0 class/category.
        
        output_type ([str, optional]): 
            String of either 'deciles' or 'quantiles' specifying if decile values or quantile  
            values are plotted. Default is 'deciles'
    
        line_color ([str, optional)]: 
            String specifying the color of the lines that will be plotted. Defaults to black.
      
        line_width ([float, optional)]: 
            Width of the lines that will divide each cell. Defaults to 1.

        title ([str, optional]): 
            Title of the generated plot. Defaults to "Lift Chart".
            
        title_fontsize ([str or int, optional]): 
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 14.
            
        text_fontsize ([string or int, optional]): 
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 10.
            
        x_ax_title ([str, optional]): 
            X-axis title of the generated plot. Defaults to "events".      
        
        x_ax_weight ([str, optional]): 
            String specifying the x-axis title weight if title is desired, choices include the default, which is 'bold', 
            'light', 'normal', 'medium', 'semibold', 'heavy', and 'black'. 
     
        y_lift_ax_title ([str, optional]): 
            Y-axis title of the generated plot. Defaults to "Lift".
                 
        y_gain_ax_title ([str, optional]): 
            Y-axis title of the generated plot. Defaults to "Gain".
        
        y_ax_weight ([str, optional]): 
            String specifying the y-axis title weight if title is desired, choices include the default, which is 'bold', 
            'light', 'normal', 'medium', 'semibold', 'heavy', and 'black'.
            
        y_lab_pad_right ([float, optional]):
            Float specifying how much to pad the y-axis title for Lift.

        ax_title_size ([int, optional]): 
            Integer for axis title font size. Default value is 12.
   
        grid_color ([str, optional]):
            String specifying the color of the lower value of the color scale. Defaults to '#EEEEEE'.

        grid_line_width ([float, optional)]: 
            Width of the grid lines. Defaults to 0.8.
            
        plot_style(string, optional): 
            Check available styles "plt.style.available".
            ['default', 'classic', 'Solarize_Light2', '_classic_test_patch', 'bmh', 'dark_background', 
             'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 
             'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 
             'seaborn-notebook', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white',
             'seaborn-whitegrid', 'tableau-colorblind10'] Defaults to `default`.
            
        round_decimal (int, optional): The decimal precision till which the result is 
            needed. Defaults to '3'.
            
        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 14.
            
        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values (8, 10, 12, etc.)
            Defaults to 10.
        
        legend_size ([str or int, optional]):        
            String {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'} or integer for the legend size. Defaults to 'small'.

        figsize ([int,int]): 
            Tuple denoting figure size of the plot, e.g. (6, 6). Defaults to `None`.
            
    Returns:
        if output_type=="decile":
        [pandas.DataFrame] The dataframe of deciles and related information.
        &
        [matplotlib.axes._subplots.AxesSubplot] Plot of cumulative gain and lift at each decile.
              
        if output_type=="quartile":
        [pandas.DataFrame] The dataframe of quartiles and related information.
        &
        [matplotlib.axes._subplots.AxesSubplot] Plot of cumulative gain and lift at each quartile.
    
    Example:
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn import tree
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=3)
        >>> clf = tree.DecisionTreeClassifier(max_depth=1,random_state=3)
        >>> clf = clf.fit(X_train, y_train)
        >>> y_prob = clf.predict_proba(X_test)
        >>> gain_lift_report(y_test, y_prob[:,0])
    """
    # Decile Gain Plot
    if output_type=="deciles":
    
        dc = gain_lift_table(y_true, y_prob, category=category, output_type="deciles", labels=labels, round_decimal=round_decimal)

        if plot_style is None:
            None
        else:
            plt.style.use(plot_style)

        # Set Plot/Subplot Dimensions  
        fig = plt.figure(figsize=figsize)

        # Cumulative Gains Plot
        plt.subplot(2, 2, 1)
        plot_gain(y_true, 
                  y_prob, 
                  category         = category, 
                  output_type      = "deciles",
                  line_color       = line_color,
                  line_width       = line_width,
                  title_fontsize   = title_fontsize,
                  title_weight     = title_weight,
                  text_fontsize    = text_fontsize,
                  x_ax_title       = x_ax_title,
                  x_ax_weight      = x_ax_weight,
                  y_ax_title       = y_gain_ax_title,
                  y_ax_weight      = y_ax_weight,
                  ax_title_size    = ax_title_size,
                  legend_size      = legend_size,
                  grid_color       = grid_color,
                  grid_line_width  = grid_line_width)

        # Cumulative Lift Plot
        plt.subplot(2, 2, 2)
        plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
        plt.tick_params(left=False, bottom=True)
        plt.ylabel(y_lift_ax_title, labelpad=y_lab_pad_right)
        plot_lift(y_true, 
                  y_prob,
                  category         = category, 
                  output_type      ="deciles",
                  line_color       = line_color,
                  line_width       = line_width,
                  title_fontsize   = title_fontsize,
                  title_weight     = title_weight,
                  text_fontsize    = text_fontsize,
                  x_ax_title       = x_ax_title,
                  x_ax_weight      = x_ax_weight,
                  y_ax_title       = y_lift_ax_title,
                  y_ax_weight      = y_ax_weight,
                  ax_title_size    = ax_title_size,
                  legend_size      = legend_size,
                  grid_color       = grid_color, 
                  grid_line_width  = grid_line_width)
        
        plt.tight_layout()

        return dc
    
    # Quartile Gain Plot
    if output_type=="quartiles":
    
        qc = gain_lift_table(y_true, y_prob, category=category, output_type="quartiles", labels=labels, round_decimal=round_decimal)

        if plot_style is None:
            None
        else:
            plt.style.use(plot_style)

        # Set Plot/Subplot Dimensions  
        fig = plt.figure(figsize=figsize)

        # Cumulative Gains Plot
        plt.subplot(2, 2, 1)
        plot_gain(y_true, 
                  y_prob, 
                  output_type      = "quartiles",
                  line_color       = line_color,
                  line_width       = line_width,
                  title_fontsize   = title_fontsize,
                  title_weight     = title_weight,
                  text_fontsize    = text_fontsize,
                  x_ax_title       = x_ax_title,
                  x_ax_weight      = x_ax_weight,
                  y_ax_title       = y_gain_ax_title,
                  y_ax_weight      = y_ax_weight,
                  ax_title_size    = ax_title_size,
                  legend_size      = legend_size,
                  grid_color       = grid_color,
                  grid_line_width  = grid_line_width)
        
        # Cumulative Lift Plot
        plt.subplot(2, 2, 2)
        plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
        plt.tick_params(left=False, bottom=True)
        plt.ylabel(y_lift_ax_title, labelpad=y_lab_pad_right)
        plot_lift(y_true,
                  y_prob, 
                  output_type      = "quartiles",
                  line_color       = line_color,
                  line_width       = line_width,
                  title_fontsize   = title_fontsize,
                  title_weight     = title_weight,
                  text_fontsize    = text_fontsize,
                  x_ax_title       = x_ax_title,
                  x_ax_weight      = x_ax_weight,
                  y_ax_title       = y_lift_ax_title,
                  y_ax_weight      = y_ax_weight,
                  ax_title_size    = ax_title_size,
                  legend_size      = legend_size,
                  grid_color       = grid_color,
                  grid_line_width  = grid_line_width)
        
        plt.tight_layout()

        return qc
