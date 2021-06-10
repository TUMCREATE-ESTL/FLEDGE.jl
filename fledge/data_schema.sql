CREATE TABLE der_cooling_plants (
    definition_name TEXT,
    plant_pump_efficiency TEXT,
    condenser_pump_head TEXT,
    evaporator_pump_head TEXT,
    chiller_set_beta TEXT,
    chiller_set_condenser_minimum_temperature_difference TEXT,
    chiller_set_evaporation_temperature TEXT,
    chiller_set_cooling_capacity TEXT,
    condenser_water_temperature_difference TEXT,
    condenser_water_enthalpy_difference TEXT,
    cooling_tower_set_reference_temperature_condenser_water TEXT,
    cooling_tower_set_reference_temperature_wet_bulb TEXT,
    cooling_tower_set_reference_temperature_slope TEXT,
    cooling_tower_set_ventilation_factor TEXT,
    PRIMARY KEY(definition_name)
);
CREATE TABLE der_ev_chargers (
    definition_name TEXT,
    nominal_charging_definition_type TEXT,
    nominal_charging_definition_name TEXT,
    maximum_charging_definition_type TEXT,
    maximum_charging_definition_name TEXT,
    maximum_discharging_definition_type TEXT,
    maximum_discharging_definition_name TEXT,
    maximum_energy_definition_type TEXT,
    maximum_energy_definition_name TEXT,
    departing_energy_definition_type TEXT,
    departing_energy_definition_name TEXT,
    PRIMARY KEY(definition_name)
);
CREATE TABLE der_models (
    der_type TEXT,
    der_model_name TEXT,
    definition_type TEXT,
    definition_name TEXT,
    power_per_unit_minimum TEXT,
    power_per_unit_maximum TEXT,
    power_factor_minimum TEXT,
    power_factor_maximum TEXT,
    energy_storage_capacity_per_unit TEXT,
    charging_efficiency TEXT,
    self_discharge_rate TEXT,
    marginal_cost TEXT,
    heat_pump_efficiency TEXT,
    thermal_efficiency TEXT,
    electric_efficiency TEXT,
    PRIMARY KEY(der_type,der_model_name)
);
CREATE TABLE der_schedules (
    definition_name TEXT,
    time_period TEXT,
    value REAL,
    PRIMARY KEY(definition_name,time_period)
);
CREATE TABLE der_timeseries (
    definition_name TEXT,
    time TEXT,
    value REAL,
    PRIMARY KEY(definition_name,time)
);
CREATE TABLE electric_grid_ders (
    electric_grid_name TEXT,
    der_name TEXT,
    der_type TEXT,
    der_model_name TEXT,
    node_name TEXT,
    is_phase_1_connected TEXT,
    is_phase_2_connected TEXT,
    is_phase_3_connected TEXT,
    connection TEXT,
    active_power_nominal TEXT,
    reactive_power_nominal TEXT,
    in_service TEXT DEFAULT 1,
    PRIMARY KEY(electric_grid_name,der_name)
);
CREATE TABLE electric_grid_line_types (
    line_type TEXT,
    n_phases TEXT,
    maximum_current TEXT,
    definition_type TEXT DEFAULT 'matrix',
    PRIMARY KEY(line_type)
);
CREATE TABLE electric_grid_line_types_matrices (
    line_type TEXT,
    row INTEGER,
    col INTEGER,
    resistance TEXT,
    reactance TEXT,
    capacitance TEXT,
    PRIMARY KEY(line_type,row,col)
);
CREATE TABLE electric_grid_line_types_overhead (
    line_type TEXT,
    phase_1_conductor_id TEXT,
    phase_2_conductor_id TEXT,
    phase_3_conductor_id TEXT,
    neutral_conductor_id TEXT,
    earth_resistivity REAL,
    air_permittivity TEXT,
    phase_1_x REAL,
    phase_1_y REAL,
    phase_2_x REAL,
    phase_2_y REAL,
    phase_3_x REAL,
    phase_3_y REAL,
    neutral_x REAL,
    neutral_y REAL,
    PRIMARY KEY(line_type)
);
CREATE TABLE electric_grid_line_types_overhead_conductors (
    conductor_id TEXT,
    conductor_size_description TEXT,
    conductor_stranding_description TEXT,
    conductor_material_description TEXT,
    conductor_diameter REAL,
    conductor_geometric_mean_radius REAL,
    conductor_resistance REAL,
    conductor_maximum_current REAL,
    PRIMARY KEY(conductor_id)
);
CREATE TABLE electric_grid_lines (
    electric_grid_name TEXT,
    line_name TEXT,
    line_type TEXT,
    node_1_name TEXT,
    node_2_name TEXT,
    is_phase_1_connected TEXT,
    is_phase_2_connected TEXT,
    is_phase_3_connected TEXT,
    length TEXT,
    in_service TEXT DEFAULT 1,
    PRIMARY KEY(electric_grid_name,line_name)
);
CREATE TABLE electric_grid_nodes (
    electric_grid_name TEXT,
    node_name TEXT,
    is_phase_1_connected TEXT,
    is_phase_2_connected TEXT,
    is_phase_3_connected TEXT,
    voltage TEXT,
    latitude TEXT,
    longitude TEXT,
    in_service TEXT DEFAULT 1,
    PRIMARY KEY(electric_grid_name,node_name)
);
CREATE TABLE electric_grid_operation_limit_types (
    electric_grid_operation_limit_type TEXT,
    voltage_per_unit_minimum TEXT,
    voltage_per_unit_maximum TEXT,
    branch_flow_per_unit_maximum TEXT,
    PRIMARY KEY(electric_grid_operation_limit_type)
);
CREATE TABLE electric_grid_transformer_types (
    transformer_type TEXT,
    resistance_percentage TEXT,
    reactance_percentage TEXT,
    tap_maximum_voltage_per_unit TEXT,
    tap_minimum_voltage_per_unit TEXT,
    PRIMARY KEY(transformer_type)
);
CREATE TABLE electric_grid_transformers (
    electric_grid_name TEXT,
    transformer_name TEXT,
    transformer_type TEXT,
    node_1_name TEXT,
    node_2_name TEXT,
    is_phase_1_connected TEXT,
    is_phase_2_connected TEXT,
    is_phase_3_connected TEXT,
    connection TEXT,
    apparent_power TEXT,
    in_service TEXT DEFAULT 1,
    PRIMARY KEY(electric_grid_name,transformer_name)
);
CREATE TABLE electric_grids (
    electric_grid_name TEXT,
    source_node_name TEXT,
    base_frequency TEXT,
    is_single_phase_equivalent TEXT DEFAULT 0,
    PRIMARY KEY(electric_grid_name)
);
CREATE TABLE parameters (
    parameter_set TEXT,
    parameter_name TEXT,
    parameter_value REAL,
    PRIMARY KEY(parameter_set,parameter_name)
);
CREATE TABLE price_timeseries (
    price_type TEXT,
    time TEXT,
    price_value REAL,
    PRIMARY KEY(price_type,time)
);
CREATE TABLE scenarios (
    scenario_name TEXT,
    electric_grid_name TEXT,
    thermal_grid_name TEXT,
    parameter_set TEXT,
    price_type TEXT,
    price_sensitivity_coefficient REAL DEFAULT 0,
    electric_grid_operation_limit_type TEXT,
    thermal_grid_operation_limit_type TEXT,
    timestep_start TEXT,
    timestep_end TEXT,
    timestep_interval TEXT,
    base_apparent_power REAL DEFAULT 1,
    base_voltage REAL DEFAULT 1,
    base_thermal_power REAL DEFAULT 1,
    PRIMARY KEY(scenario_name)
);
CREATE TABLE thermal_grid_ders (
    thermal_grid_name TEXT,
    der_name TEXT,
    node_name TEXT,
    der_type TEXT,
    der_model_name TEXT,
    thermal_power_nominal TEXT,
    in_service TEXT DEFAULT 1,
    PRIMARY KEY(thermal_grid_name,der_name)
);
CREATE TABLE thermal_grid_line_types (
    line_type TEXT,
    diameter TEXT,
    absolute_roughness TEXT,
    maximum_velocity TEXT,
    PRIMARY KEY(line_type)
);
CREATE TABLE thermal_grid_lines (
    thermal_grid_name TEXT,
    line_name TEXT,
    line_type TEXT,
    node_1_name TEXT,
    node_2_name TEXT,
    length TEXT,
    in_service TEXT DEFAULT 1,
    PRIMARY KEY(thermal_grid_name,line_name)
);
CREATE TABLE thermal_grid_nodes (
    thermal_grid_name TEXT,
    node_name TEXT,
    node_type TEXT,
    latitude TEXT,
    longitude TEXT,
    in_service TEXT DEFAULT 1,
    PRIMARY KEY(thermal_grid_name,node_name)
);
CREATE TABLE thermal_grid_operation_limit_types (
    thermal_grid_operation_limit_type TEXT,
    node_head_per_unit_maximum TEXT,
    pipe_flow_per_unit_maximum TEXT,
    PRIMARY KEY(thermal_grid_operation_limit_type)
);
CREATE TABLE thermal_grids (
    thermal_grid_name TEXT,
    source_node_name TEXT,
    distribution_pump_efficiency TEXT,
    energy_transfer_station_head_loss TEXT,
    enthalpy_difference_distribution_water TEXT,
    water_density TEXT,
    water_kinematic_viscosity TEXT,
    plant_type TEXT,
    plant_model_name TEXT,
    PRIMARY KEY(thermal_grid_name)
);
