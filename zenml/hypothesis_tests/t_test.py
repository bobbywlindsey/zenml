from scipy.stats import ttest_ind, ttest_rel
from ..utils import is_valid_alt_hypothesis, get_p_value, effect_size, get_power


def t_test(alt_hypothesis, intervention_data,
           no_intervention_data, variance_assumption='equal', 
           is_paired=False):
    """
    :param alt_hypothesis: str
    :param intervention_data: pandas.Series
    :param no_intervention_data: pandas.Series
    :param variance_assumption: str
    :param is_paired: boolean
    :return: dict
    """
    test_results = {}
    n1 = len(intervention_data)
    n2 = len(no_intervention_data)
    test_results['Sample Size 1'] = n1
    test_results['Sample Size 2'] = n2
    test_results['Alt. Hypothesis'] = alt_hypothesis
    
    if is_valid_alt_hypothesis(alt_hypothesis):
        # Welch's t-test
        if not is_paired and (n1 != n2 or variance_assumption == 'unequal'):
            test_statistic, p_value = ttest_ind(intervention_data, no_intervention_data,
                                                equal_var=False)

            test_results['Test Name'] = 'Welchs t-test'
        # Student's t-test
        elif not is_paired:
            test_statistic, p_value = ttest_ind(intervention_data, no_intervention_data)
            test_results['Test Name'] = 'Students t-test'
        # Paired t-test
        elif is_paired:
            # check to make sure equal length of data
            if n1 != n2:
                raise ValueError('Sample sizes must be equal to each other')
            test_statistic, p_value = ttest_rel(intervention_data, no_intervention_data)
            test_results['Test Name'] = 'Paired t-test'
        
        test_results['p-value'] = p_value
        test_results['Test Statistic'] = test_statistic
        # get p-value
        p_value = get_p_value(test_results)
        test_results['p-value'] = p_value
        # get effect size
        effect = effect_size(test_results['Test Name'], intervention_data,
                             no_intervention_data, 'independent')
        test_results.update(effect)
        # get the power
        power = get_power(test_results)
        test_results['Power'] = power
        
    return test_results