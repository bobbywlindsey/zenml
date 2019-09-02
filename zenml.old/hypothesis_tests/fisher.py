from scipy.stats import fisher_exact
from ..utils import is_valid_alt_hypothesis, get_power


def fisher(alt_hypothesis, intervention_data, no_intervention_data):
    """
    :param alt_hypothesis: str
    :param intervention_data: pandas.Series
    :param no_intervention_data: pandas.Series
    :return: dict
    """
    test_results = {}
    n1 = len(intervention_data)
    n2 = len(no_intervention_data)
    test_results['Sample Size 1'] = n1
    test_results['Sample Size 2'] = n2
    test_results['Alt. Hypothesis'] = alt_hypothesis

    # get summary stats from both sites
    no_intervention_data_clicks = no_intervention_data.sum()
    no_intervention_data_no_clicks = no_intervention_data[no_intervention_data == 0].shape[0]
    intervention_data_clicks = intervention_data.sum()
    intervention_data_no_clicks = intervention_data[intervention_data == 0].shape[0]

    # create contingency table
    contingency_table = [[no_intervention_data_no_clicks, intervention_data_no_clicks],
                         [no_intervention_data_clicks, intervention_data_clicks]]
    
    if is_valid_alt_hypothesis(alt_hypothesis):
        if alt_hypothesis == '!=':
            odds_ratio, p_value = fisher_exact(contingency_table, alternative='two-sided')
        elif alt_hypothesis == '>':
            odds_ratio, p_value = fisher_exact(contingency_table, alternative='greater')
        elif alt_hypothesis == '<':
            odds_ratio, p_value = fisher_exact(contingency_table, alternative='less')
        test_results['Test Name'] = 'Fishers Exact Test'
        test_results['p-value'] = p_value
        test_results['Effect Size'] = odds_ratio/8.3 # to convert odds ratio to Cohen's D
        
        # get power (currently not implemented)
        power = get_power(test_results)
        test_results['Power'] = power
        
    return test_results