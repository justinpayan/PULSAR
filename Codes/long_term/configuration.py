import os
configurable_dict = dict(
    main=os.path.join("Files", "Cycle 10", "sb12m_master_prepared_c10.csv"),
    availability=os.path.join("Files", "Cycle 10", "expected_times_c10_for_strategic.csv"),
    simulation=os.path.join("Files", "Cycle 10", "sb_12m_pressure_new.csv"),
    modes=os.path.join("Files", "Cycle 10", "sb12m_master_with_modes.csv"),
    year='2023',
    cycle_accepted=os.path.join("Files", "Cycle 10", 'accepted_projects_cycle10.csv'),
    borderconf_1='3A',
    borderconf_2='1A',
    d=0,
    lambda_p=0.4,
    total_bins=8600,
    model_gap=0.01,
    # problem="assign",
    # out_file=os.path.join("Files", "Cycle 9", "Planning Problem", 'accepted_projects_cycle9.csv'),
    problem='plan',
    out_file=os.path.join("Files", "Cycle 10", "cycle10_med_term.csv"),
    project_weight=100,
    # set_date='2020-11-28',
    # completed_sbs=[0] #list_sb_iuds_completed
)