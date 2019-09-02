import math
from scipy.stats import pearsonr, linregress
from statsmodels.stats.power import TTestIndPower


def is_valid_alt_hypothesis(alt_hypothesis):
    """
    :param alt_hypothesis: str
    :return: boolean
    """
    # check for valid alt_hypothesis
    if alt_hypothesis not in ('!=', '>', '<'):
        raise ValueError('alt_hypothesis value not valid: try !=, >, or < instead')
    return True


def get_p_value(test_results):
    """
    :param test_results: dict 
    :return: float
    """
    test_name = test_results['Test Name']
    alt_hypothesis = test_results['Alt. Hypothesis']
    p_value = test_results['p-value']
    test_statistic = test_results['Test Statistic']
    if 't-test' in test_name:
        if alt_hypothesis == '!=':
            return p_value
        elif alt_hypothesis == '>':
            return p_value/2
            if test_statistic > 0 and p_value/2 < 0.05:
                print('Reject null hypothesis')
            else:
                print('Fail to reject null hypothesis')
        elif alt_hypothesis == '<':
            return p_value/2
            if test_statistic < 0 and p_value/2 < 0.05:
                print('Reject null hypothesis')
            else:
                print('Fail to reject null hypothesis')


def cohens_d(intervention_data, no_intervention_data, collection_method='independent'):
    """
    :param intervention_data: pandas.Series
    :param no_intervention_data: pandas.Series
    :param collection_method: str
    :return: dict
    """
    n1 = len(intervention_data)
    n2 = len(no_intervention_data)
    u1 = intervention_data.mean()
    u2 = no_intervention_data.mean()
    sigma1 = intervention_data.std()
    sigma2 = no_intervention_data.std()
    
    pooled_est_std = math.sqrt(((n1 - 1) * sigma1**2 + (n2 - 1) * sigma2**2) / (n1 + n2 - 2))
    cohens_d = (u1 - u2) / pooled_est_std
    
    return {'Effect Size': cohens_d}


def effect_size(test_name, intervention_data, no_intervention_data,
                collection_method='independent'):
    """
    :param test_name: str
    :param intervention_data: pandas.Series
    :param no_intervention_data: pandas.Series
    :param collection_method: str
    :return: dict
    """
    n1 = len(intervention_data)
    n2 = len(no_intervention_data)
    effect_sizes = {}
    # effect sizes measured by association
    if n1 == n2 and collection_method=='independent':
        correlation_coefficient, _ = pearsonr(intervention_data, no_intervention_data)
        effect_sizes['Effect Size'] = correlation_coefficient
        slope, intercept, r_value, p_value, std_err = linregress(intervention_data, no_intervention_data)
        effect_sizes['Effect Size': r_value**2]
    # effect sizes measure by difference between variables
    else:
        if test_name == 'Students t-test' or test_name == 'Welchs t-test':
            effect_sizes.update(cohens_d(intervention_data, no_intervention_data, collection_method))
        elif test_name == 'Paired t-test':
            effect_sizes.update(cohens_d(intervention_data, no_intervention_data, collection_method))
    return effect_sizes
    # if assumed distribution is binomial, then report odds ratio using Fisher's Exact test
    # and relative risk ratio; both ratios rely on contigency table and this is done in the t_test module

def find_power_of_hypothesis_test(test_name, alt_hypothesis, significance_level,
                                  effect_size, sample_size1, ratio=1.0):
    """
    :param test_name: str
    :param alt_hypothesis: str
    :param significance_level: float
    :param effect_size: float
    :param sample_size1: int
    :param ration: float
    :return: float or none
    """
    if 'Fisher' not in test_name:
        # perform power analysis
        analysis = TTestIndPower()
        if alt_hypothesis == '!=':
            power = analysis.power(effect_size, nobs1=sample_size1,
                                   alpha=significance_level, ratio=ratio)
        elif alt_hypothesis == '<':
            power = analysis.power(effect_size, nobs1=sample_size1,
                                   alpha=significance_level, ratio=ratio,
                                   alternative='smaller')
        elif alt_hypothesis == '>':
            power = analysis.power(effect_size, nobs1=sample_size1,
                                   alpha=significance_level, ratio=ratio,
                                   alternative='larger')
        return power
    else:
        return None


def get_power(test_results):
    """
    :param test_results: dict
    :return: float
    """
    test_name = test_results['Test Name']
    alpha = test_results['p-value']
    effect = test_results['Effect Size']
    sample_size1 = test_results['Sample Size 1']
    ratio = test_results['Sample Size 2'] / sample_size1
    alt_hypothesis = test_results['Alt. Hypothesis']
    power = find_power_of_hypothesis_test(test_name, alt_hypothesis,
                                          alpha, effect, sample_size1, ratio)
    return power