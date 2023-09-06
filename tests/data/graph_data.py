class InterventionData:
    # tests .do() method raises error
    # inputs: outline
    do_erroneous_data = [
        # test non-compliance support for bernoulli
        ({'A': 'bernoulli(p_=0.2)'}),
        # test non-compliance support for poisson
        ({'A': 'poisson(lambda_=1)'}),
        # test non-compliance support for exponential
        ({'A': 'exponential(lambda_=1)'})
    ]

    # tests .do_functional() method with 1 parent
    # inputs: outline, parent, child
    functional_do_data = [
        # normal scenario: A is already a parent of B
        ({'A': 'normal(mu_=0, sigma_=1)',
          'B': 'normal(mu_=A+1, sigma_=1)'}, 'A', 'B'),
        # scenario: A is not a parent of B, same topo rank
        ({'A': 'normal(mu_=0, sigma_=1)',
          'B': 'normal(mu_=0, sigma_=1)'}, 'A', 'B'),
        # scenario: A and B same rank, but B parent of A
        ({'A': 'normal(mu_=0, sigma_=1)',
          'B': 'normal(mu_=0, sigma_=1)'}, 'B', 'A'),
        # scenario: A in lower rank as B, B parent of A
        ({'B': 'normal(mu_=0, sigma_=1)',
          'C': 'normal(mu_=0, sigma_=1)',
          'A': 'normal(mu_=2C, sigma_=1)'}, 'B', 'A'),
        # scenario: A lower rank than B, not descendant of B
        ({'B': 'normal(mu_=0, sigma_=1)',
          'C': 'normal(mu_=0, sigma_=1)',
          'A': 'normal(mu_=2C, sigma_=1)'}, 'A', 'B')
    ]

    # tests .do_functional() method with 2 parents
    # inputs: outline, parents, child
    functional_do_two_parent_data = [
        ({'A': 'normal(mu_=0, sigma_=1)',
          'B': 'normal(mu_=A+1, sigma_=1)',
          'C': 'constant(3)'}, ['A', 'C'], 'B')
    ]

    # tests .do_functional() method raises error due to loops
    # inputs: outline, parent, child
    functional_do_loop_error_data = [
        ({'A': 'normal(mu_=0, sigma_=1)',
          'B': 'normal(mu_=A+1, sigma_=1)'}, 'B', 'A'),
        ({'A': 'normal(mu_=0, sigma_=1)',
          'B': 'normal(mu_=A+1, sigma_=1)',
          'C': 'normal(mu_=B, sigma_=1)'}, 'C', 'A')
    ]

    # tests .do_functional() and .do_self() methods raises error due to support non-compliance
    # inputs: outline
    support_error_data = [
        ({'A': 'bernoulli(p_=0.2)', 'B': 'bernoulli(p_=0.2)'}),
        # non-compliance support for poisson
        ({'A': 'poisson(lambda_=1)', 'B': 'bernoulli(p_=0.2)'}),
        # non-compliance support for exponential
        ({'A': 'exponential(lambda_=1)', 'B': 'bernoulli(p_=0.2)'})
    ]