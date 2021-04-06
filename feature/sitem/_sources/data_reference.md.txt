# Data reference

``` warning::
    This reference is work in progress.
```

FLEDGE scenarios are defined through CSV files, where each CSV file represents a table as defined below (the file name is interpreted as the table name). Internally, FLEDGE loads all CSV files into a local SQLITE database for more convenient processing. The default location for FLEDGE scenario definitions is in the `data` directory in the repository and all CSV files in the `data` directory are automatically loaded into the database. The CSV files may be structured into sub-directories, but all files are eventually combined into the same database. Hence, all type / element identifiers must be unique across all scenario definitions.

## Scenario data

### `scenarios`

Scenario definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `scenario_name` | | Unique scenario identifier.|
| `electric_grid_name` | | Electric grid identifier as defined `electric grids` |
| `thermal_grid_name` | | Thermal grid identifier as defined `thermal grids` |
| `parameter_set` | | Parameter set identifier as defined in `parameters` |
| `price_type` | | Type identifier as defined in `price_timeseries` |
| `price_sensitivity_coefficient` | $/kWh² | Price sensitivity coefficient for the quadratic cost terms of DERs / grids. Optional column, which defaults to `0` if not explicitly defined. |
| `electric_grid_operation_limit_type` | | Operation limit type as defined in `electric_grid_operation_limit_types` |
| `thermal_grid_operation_limit_type` | | Type identifier as defined in `thermal_grid_operation_limit_types` |
| `timestep_start` | | Start timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `timestep_end` | | End timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `timestep_interval` | | Time interval in format `HH:MM:SS` |
| `base_apparent_power` | W | Normalization / scaling factor¹ for apparent / active / reactive power variables. Optional column, which defaults to `1` if not explicitly defined. |
| `base_voltage` | W | Normalization / scaling factor¹ for voltage variables. Optional column, which defaults to `1` if not explicitly defined. |
| `base_thermal_power` | W | Normalization / scaling factor¹ for voltage variables. Optional column, which defaults to `1` if not explicitly defined. |

¹ Normalization / scaling factors are used internally to reduce the order of magnitude of variable values. Appropriate scaling significantly improves numerical performance, especially in numerical optimization problems. Note that this only applies internally to variables and models, but inputs and outputs are reported unscaled in SI units.

### `parameters`

In all tables, a `parameter_name` string can be used to define numerical parameters in place of numerical values (identifiers or string values cannot be replaced with parameters). During model setup, those strings will be parsed from the `parameters` table to obtain the corresponding numerical values.

| Column | Unit | Description |
| --- |:---:| --- |
| `parameter_set` | | Parameter set identifier. |
| `parameter_name` | | Unique parameter identifier (must only be unique within the associated parameter set).|
| `parameter_value` | - | Parameter value. |

### `price_timeseries`

Price timeseries.

| Column | Unit | Description |
| --- |:---:| --- |
| `price_type` | | Unique type identifier.|
| `time` | | Timestamp according to ISO 8601. |
| `price_value` | $/kWh | Price value. *Currently, prices / costs are assumed to be in SGD.* |

## Electric grid data

### `electric_grids`

Electric grid definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `electric_grid_name` | | Unique electric grid identifier. |
| `source_node_name` | | Source node name as defined in `electric_grid_nodes` |
| `base_frequency` | Hz | Nominal grid frequency. |
| `is_single_phase_equivalent` | | Single-phase-equivalent modelling flag¹. If `0`, electric grid is modelled as multi-phase system. If `1`, electric grid is modelled as single-phase-equivalent of a three-phase balanced system. Optional column, which defaults to `0` if not explicitly defined. |

¹ If single-phase-equivalent modelling is used, all nodes, lines, transformers and DERs must be defined as single-phase elements, i.e., these elements should be connected only to phase 1. However, all power values (DER active / reactive power, transformer apparent power) must be defined as total three-phase power.

### `electric_grid_ders`

Distributed energy resources (DERs) in the electric grid. Can define both loads (negative power) and generations (positive power). The corresponding DER models are defined in `der_models`. The selection of DER types will be extended in the future.

| Column | Unit | Description |
| --- |:---:| --- |
| `electric_grid_name` | | Electric grid identifier as defined in `electric_grids`. |
| `der_name` | | Unique DER identifier (must only be unique within the associated electric grid). |
| `der_type` | | DER type selector, which determines the type of DER model to be used. Choices: `fixed_load`, `flexible_load`, `fixed_ev_charger`, `flexible_building`, `fixed_generator`, `flexible_generator`, `cooling_plant`. |
| `der_model_name` | | DER model identifier as defined in `der_models`. For `flexible_building`, this defines the CoBMo the scenario name. |
| `node_name` | | Node identifier as defined in `electric_grid_nodes`. |
| `is_phase_1_connected` | | Selector for connection at phase 1. Choices: `0` (connected), `1` (not connected). |
| `is_phase_2_connected` | | Selector for connection at phase 2. Choices: `0` (connected), `1` (not connected). |
| `is_phase_3_connected` | | Selector for connection at phase 3. Choices: `0` (connected), `1` (not connected). |
| `connection` | | Selector for Wye / Delta connection. Choices: `wye`, `delta`. |
| `active_power_nominal` | W | Nominal active power, where loads are negative and generations are positive. |
| `reactive_power_nominal` | VAr | Nominal reactive power, where loads are negative and generations are positive. |
| `in_service` | | In-service selector. Not-in-service grid elements are ignored and not loaded into the model. Choices: `1` (in service) or `0` (not in service). Optional column, which defaults to `1` if not explicitly defined. |

### `electric_grid_line_types`

Electric line type definitions are split into `electric_grid_line_types` for the general type definition and `electric_grid_line_types_matrices` for the definition of electric characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `line_type` | | Unique type identifier. |
| `n_phases` | - | Number of phases. |
| `maximum_current` | A | Maximum permissible current (thermal line limit). |

### `electric_grid_line_types_matrices`

Electric line characteristics are defined in terms of element property matrices. Note that the matrices are expected to be symmetric and therefore only the lower triangular matrix should be defined. The matrices are defined element-wise (indexed by row / column pairs), to allow definition of single-phase line types alongside multi-phase line types.

| Column | Unit | Description |
| --- |:---:| --- |
| `line_type` | | Line type identifier as defined in `electric_grid_line_types`. |
| `row` | | Element matrix row number (first row is `1`). |
| `col` | | Element matrix column number (first column is `1`). |
| `resistance` | Ω/km | Series resistance matrix entry. |
| `reactance` | Ω/km | Series reactance matrix entry. |
| `capacitance` | nF/km | Shunt capacitance matrix entry. |

### `electric_grid_lines`

Electric grid lines.

| Column | Unit | Description |
| --- |:---:| --- |
| `electric_grid_name` | | Electric grid identifier as defined in `electric_grids` |
| `line_name` | | Unique line identifier (must only be unique within the associated electric grid). |
| `line_type` | | Line type identifier as defined in `electric_grid_line_types`. |
| `node_1_name` | | Start node identifier as defined in `electric_grid_nodes` |
| `node_2_name` | | End node identifier as defined in `electric_grid_nodes` |
| `is_phase_1_connected` | | Selector for connection at phase 1. Choices: `0` (connected), `1` (not connected). |
| `is_phase_2_connected` | | Selector for connection at phase 2. Choices: `0` (connected), `1` (not connected). |
| `is_phase_3_connected` | | Selector for connection at phase 3. Choices: `0` (connected), `1` (not connected). |
| `length` | km | Line length. |
| `in_service` | | In-service selector. Not-in-service grid elements are ignored and not loaded into the model. Choices: `1` (in service) or `0` (not in service). Optional column, which defaults to `1` if not explicitly defined. |

### `electric_grid_nodes`

Electric grid nodes.

| Column | Unit | Description |
| --- |:---:| --- |
| `electric_grid_name` | | Electric grid identifier as defined in `electric_grids` |
| `node_name` | | Unique node identifier (must only be unique within the associated electric grid). |
| `is_phase_1_connected` | | Selector for connection at phase 1. Choices: `0` (connected), `1` (not connected). |
| `is_phase_2_connected` | | Selector for connection at phase 2. Choices: `0` (connected), `1` (not connected). |
| `is_phase_3_connected` | | Selector for connection at phase 3. Choices: `0` (connected), `1` (not connected). |
| `voltage` | V | Nominal voltage. |
| `latitude` | | Latitude. |
| `longitude` | | Longitude. |
| `in_service` | | In-service selector. Not-in-service grid elements are ignored and not loaded into the model. Choices: `1` (in service) or `0` (not in service). Optional column, which defaults to `1` if not explicitly defined. |

### `electric_grid_operation_limit_types`

Operation limit type definition for the electric grid. This information is utilized for the definition of the operational constraints in an optimal operation problem.

| Column | Unit | Description |
| --- |:---:| --- |
| `electric_grid_operation_limit_type` | | Unique type identifier. |
| `voltage_per_unit_minimum` | - | Minimum voltage in per unit of the nominal voltage. |
| `voltage_per_unit_maximum` | - | Maximum voltage in per unit of the nominal voltage. |
| `branch_flow_per_unit_maximum` | - | Maximum branch flow in per unit of the branch flow at nominal loading conditions. |

### `electric_grid_transformer_types`

Transformer type characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `transformer_type` | | Unique type identifier. |
| `resistance_percentage` | - | Resistance percentage. |
| `reactance_percentage` | - | Reactance percentage. |
| `tap_maximum_voltage_per_unit` | - | Maximum secondary side tap position. (Currently not used.) |
| `tap_minimum_voltage_per_unit` | - | Minimum secondary side tap position. (Currently not used.) |

### `electric_grid_transformers`

Electric grid transformers, which are limited to transformers with two windings, where the same number of phases is connected at each winding.

| Column | Unit | Description |
| --- |:---:| --- |
| `electric_grid_name` | | Electric grid identifier as defined in `electric_grids` |
| `transformer_name` | | Unique transformer identifier (must only be unique within the associated electric grid).|
| `transformer_type` | | Transformer type identifier as defined in `electric_grid_transformer_types` |
| `node_1_name` | | Primary side node name as defined in `electric_grid_nodes` |
| `node_2_name` | | Secondary side node name as defined in `electric_grid_nodes` |
| `is_phase_1_connected` | | Selector for connection at phase 1. Choices: `0` (connected), `1` (not connected). |
| `is_phase_2_connected` | | Selector for connection at phase 2. Choices: `0` (connected), `1` (not connected). |
| `is_phase_3_connected` | | Selector for connection at phase 3. Choices: `0` (connected), `1` (not connected). |
| `connection` | | Selector for Wye / Delta connection. Choices: `wye`, `delta`. Note that Wye-connected windings are assumed to be grounded. |
| `apparent_power` | VA | Nominal apparent power loading. |
| `in_service` | | In-service selector. Not-in-service grid elements are ignored and not loaded into the model. Choices: `1` (in service) or `0` (not in service). Optional column, which defaults to `1` if not explicitly defined. |

## Thermal grid data

### `thermal_grids`

Thermal grid definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | Unique thermal grid identifier. |
| `source_node_name` | | Source node name as defined in `thermal_grid_nodes`. |
| `distribution_pump_efficiency` | - | Pump efficiency (pump power / electric power) of the secondary side pumps, i.e. the pumps in the distribution system / thermal grid. |
| `energy_transfer_station_head_loss` | m | Pump pressure head loss in the energy transfer station at each DER. |
| `enthalpy_difference_distribution_water` | J/kg | Enthalpy difference between supply and return side of the distribution water, i.e. the water flowing to the thermal grid. |
| `water_density` | kg/m³ | Density of the distribution water. |
| `water_kinematic_viscosity` | m²/s | Kinematic viscosity of the distribution water. |
| `plant_type` | | Thermal supply plant type. Currently only `cooling_plant` is supported. |
| `plant_model_name` | | Plant model identifier. If plant type `cooling_plant`, as defined in `der_cooling_plants`. |

### `thermal_grid_ders`

Distributed energy resources (DERs) in the thermal grid. Can define both loads (negative power) and generations (positive power). The corresponding DER models are defined in `der_models`. The selection of DER types will be extended in the future.

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | Thermal grid identifier as defined in `thermal_grids`. |
| `der_name` | | Unique DER identifier (must only be unique within the associated thermal grid). |
| `node_name` | | Node identifier as defined in `thermal_grid_nodes`. |
| `der_type` | | DER type selector, which determines the type of DER model to be used. Choices: `flexible_building`, `fixed_generator`, `flexible_generator`, `cooling_plant`.  |
| `der_model_name` | | DER model identifier as defined in `der_models`. For `flexible_building`, this defines the CoBMo the scenario name. |
| `thermal_power_nominal` | W | Nominal thermal power, where loads are negative and generations are positive. |
| `in_service` | | In-service selector. Not-in-service grid elements are ignored and not loaded into the model. Choices: `1` (in service) or `0` (not in service). Optional column, which defaults to `1` if not explicitly defined. |

### `thermal_grid_line_types`

Thermal line types for defining pipe characteristics.

| Column | Unit | Description |
| --- |:---:| --- |
| `line_type` | | Unique type identifier. |
| `diameter` | m | Pipe diameter. |
| `absolute_roughness` | mm | Absolute roughness of the pipe. |
| `maximum_velocity` | m/s | Nominal maximum pipe velocity. |

### `thermal_grid_lines`

Thermal grid line (pipe) definitions. The definition only includes the supply side piping, as the return side is assumed be symmetric.

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | Thermal grid identifier as defined in `thermal_grids`. |
| `line_name` | | Unique line identifier (must only be unique within the associated thermal grid). |
| `line_type` | | Line type identifier as defined in `thermal_grid_line_types`. |
| `node_1_name` | | Start node identifier as defined in `thermal_grid_nodes` |
| `node_2_name` | | End node identifier as defined in `thermal_grid_nodes`. |
| `length` | km | Line length. |
| `in_service` | | In-service selector. Not-in-service grid elements are ignored and not loaded into the model. Choices: `1` (in service) or `0` (not in service). Optional column, which defaults to `1` if not explicitly defined. |

### `thermal_grid_nodes`

Thermal grid nodes.

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_name` | | Thermal grid identifier as defined in `thermal_grids`. |
| `node_name` | | Unique node identifier (must only be unique within the associated thermal grid). |
| `latitude` | | Latitude. |
| `longitude` | | Longitude. |
| `in_service` | | In-service selector. Not-in-service grid elements are ignored and not loaded into the model. Choices: `1` (in service) or `0` (not in service). Optional column, which defaults to `1` if not explicitly defined. |

### `thermal_grid_operation_limit_types`

Operation limit type definition for the thermal grid. This information is utilized for the definition of the operational constraints in an optimal operation problem. Note that thermal line limits are currently defined in per unit of the nominal thermal power solution, i.e., the thermal power flow solution for nominal loading conditions as defined in `thermal_grid_ders`, but this should be changed in future.

| Column | Unit | Description |
| --- |:---:| --- |
| `thermal_grid_operation_limit_type` | | Unique type identifier. |
| `node_head_per_unit_maximum` | - | Maximum node head, in per unit of the nominal thermal power solution. |
| `pipe_flow_per_unit_maximum` | - | Maximum pipe / branch flow, in per unit of the nominal thermal power solution. |

## Distributed energy resource (DER) data

### `der_models`

DER model parameter definitions. This table incorporates the definition of various DER types, which have different characteristics and require a different subset of the columns. See below for a detailed description of each DER type.

| Column | Unit | Description |
| --- |:---:| --- |
| `der_type` | | DER type selector. Choices: `fixed_load`, `flexible_load`, `fixed_generator`, `flexible_generator`, `fixed_ev_charger`, `cooling_plant`, `storage`. Note: `flexible_buildings` cannot be defined here, because it is obtained from the CoBMo submodule directly. |
| `der_model_name` | | Unique DER model identifier (must only be unique within the associated DER type). |
| `definition_type` | | Definition type selector, because most DER types require either additional timeseries / schedule definition¹ or other supplementary parameter definitions from either of the tables `der_timeseries`, `der_schedules` or `der_cooling_plants`. Choices: `timeseries` (Defines timeseries of absolute values².) `schedule` (Defines schedule of absolute values².), `timeseries_per_unit` (Define timeseries of per unit values².), `schedule_per_unit` (Defines schedule of per unit values².), `cooling_plant` (Defines cooling plant.) |
| `definition_name` | | Definition identifier, which corresponds to `definition_name` in either `der_timeseries`, `der_schedules` or `der_cooling_plants`. If `definition_type` is `timeseries` or `timeseries_per_unit`: defined in `der_timeseries`; if `definition_type` is `schedule` or `schedule_per_unit`: defined in `der_schedules`; if `definition_type` is `cooling_plant`: defined in `der_cooling_plants`. |
| `power_per_unit_minimum` | - | Minimum permitted power (load or generation) in per unit of the nominal power. |
| `power_per_unit_maximum` | - | Maximum permitted power (load or generation) in per unit of the nominal power. |
| `power_factor_minimum` | | Minimum permitted power factor. *Currently not used.* |
| `power_factor_maximum` | | Maximum permitted power factor. *Currently not used.* |
| `energy_storage_capacity_per_unit` | h | Energy storage capacity in per unit of the nominal active or thermal power. For example, nominal power of 1000 W and per-unit energy storage capacity of 3 h correspond to 3000 Wh energy storage capacity. |
| `charging_efficiency` | - | Energy storage charging efficiency factor. |
| `self_discharge_rate` | 1/h | Energy storage self discharge rate. |
| `marginal_cost` | $/kWh | Marginal cost of power generation. *Currently, prices / costs are assumed to be in SGD.* |

For most DER types, the `der_models` table is supplemented by timeseries / schedule definitions in the tables `der_timeseries` / `der_schedules` or supplementary parameter definitions in `der_cooling_plants` based on the columns `definition_type` / `definition_name`. Furthermore, each DER type relies on a different subset of columns / parameters in `der_models`. The table below outlines the required supplementary definitions for as well as the required columns for each DER type:

| DER type | Description | Required columns | Required supplementary definitions |
| --- | --- | --- | --- |
| `fixed_load` | Fixed load, following a fixed demand timeseries. | `definition_type`, `definition_name` | Timeseries / schedule¹ for nominal active / reactive / thermal power². |
| `flexible_load` | Flexible load, following a demand timeseries, but able shift a share of its nominal load, limited by its energy storage capacity. | `definition_type`, `definition_name`, `power_per_unit_minimum`, `power_per_unit_maximum`, `energy_storage_capacity_per_unit` | Timeseries / schedule¹ for nominal active / reactive / thermal power². |
| `fixed_generator` | Fixed generator, following a fixed generation timeseries. | `definition_type`, `definition_name`, `marginal_cost` | Timeseries / schedule¹ for nominal active / reactive / thermal power². |
| `flexible_generator` | Flexible generator, dispatchable within given limits and based on a generation timeseries. | `definition_type`, `definition_name`, `power_per_unit_minimum`, `power_per_unit_maximum`, `marginal_cost` | Timeseries / schedule¹ for nominal active / reactive / thermal power². |
| `fixed_ev_charger` | Fixed EV charger, following a fixed demand timeseries. | `definition_type`, `definition_name` | Timeseries / schedule¹ for nominal active / reactive / thermal power². |
| `cooling_plant` | Cooling plant, converts electric power to thermal power, dispatchable with nominal power limits. | `definition_type`, `definition_name` | Cooling plant parameters according to `der_cooling_plants`. |
| `storage` | Energy storage, can charge / discharge within given limits and based on its energy storage capacity. | `power_per_unit_minimum`, `power_per_unit_maximum`, `energy_storage_capacity_per_unit`, `charging_efficiency`, `self_discharge_rate` | N.A. |

The selection of DER types will be extended in the future. Note that the DER type `flexible_buildings` is not defined here, instead the model definition is obtained from the Control-oriented Building Model (CoBMo) submodule.

Not all DER types can be connected to all grid types, e.g. `fixed_ev_charger` is only available in the electric grid. Refer to `electric_grid_ders` / `thermal_grid_ders` to check which DER types can be connected respectively.

¹ For DER types which require the definition of timeseries values, these can be defined either directly as a timeseries in `der_timeseries` or as a schedule in `der_schedules`, where the latter describes recurring schedules based on weekday / time of day (see `der_schedules`).

² Active / reactive / thermal power values can be defined as absolute values or in per unit values. Per unit values are assumed to be in per unit of the nominal active / reactive power as defined `electric_grid_ders` / `thermal_grid_ders`. Note that the sign of the active / reactive / thermal power values in the timeseries / schedule definition are ignored and superseded by the sign of the nominal active / reactive / thermal power value as defined in `electric_grid_ders` / `thermal_grid_ders`, where positive values are interpreted as generation and negative values as consumption. Additionally, note that `der_timeseries` / `der_schedules` only define a single power value for each timestep. Thus, for electric DERs the active power is derived directly based on the value in `der_timeseries` / `der_schedules` and the reactive power is calculated from the active power assuming a fixed power factor according to the nominal active / reactive power in `electric_grid_ders`.

### `der_timeseries`

DER timeseries definition.

| Column | Unit | Description |
| --- |:---:| --- |
| `definition_name` | | Unique definition identifier. Corresponds to `definition_name` in `der_models`. |
| `time` | | Timestep in format `yyyy-mm-ddTHH:MM:SS` (according to ISO 8601). |
| `value` | - | Power value (absolute or per unit according to `der_models`). |

### `der_schedules`

DER schedules definition. The timeseries is constructed by obtaining the appropriate values based on the `time_period` in `ddTHH:MM` format. Each value is kept constant at the given value for any daytime greater than or equal to `HH:MM` and any weekday greater than or equal to `dd` until the next defined `ddTHH:MM`. Note that the daily schedule is repeated for any weekday greater than or equal to `dd` until the next defined `dd`. The initial value for each schedule must start at `time_period = 01T00:00`.

| Column | Unit | Description |
| --- |:---:| --- |
| `definition_name` | | Unique definition identifier. Corresponds to `definition_name` in `der_models`. |
| `time_period` | | Time period in `ddTHH:MM` format. `dd` is the weekday (`01` - Monday ... `07` - Sunday). `T` is the divider for date and time information according to ISO 8601. `HH:MM` is the daytime. |
| `value` | - | Power value (absolute or per unit according to `der_models`). |

### `der_cooling_plants`

Supplementary cooling plant model parameter definition. The cooling plant model can represent district cooling plants as well as distributed cooling plants. Cooling plants must be connected to both electric and thermal grid, therefore must be defined both in `electric_grid_ders` and `thermal_grid_ders`.

| Column | Unit | Description |
| --- |:---:| --- |
| `definition_name` | | Unique definition identifier. Corresponds to `definition_name` in `der_models`. |
| `cooling_efficiency` | | Coefficient of performance (COP). |
| `plant_pump_efficiency` | - | Pump efficiency (pump power / electric power) of the primary side pumps, i.e. the pumps within the district cooling plant. |
| `condenser_pump_head` | m | Pump pressure head across the condenser. |
| `evaporator_pump_head` | m | Pump pressure head across the evaporator. |
| `chiller_set_beta` | - | Chiller set model beta factor, used to model the chiller efficiency. |
| `chiller_set_condenser_minimum_temperature_difference` | K | Chiller set minimum temperature difference at the condenser, i.e. between the condenser water cycle and chiller refrigerant cycle. |
| `chiller_set_evaporation_temperature` | K | Chiller set evaporation temperature. |
| `chiller_set_cooling_capacity` | W | Chiller nominal maximum cooling capacity. |
| `condenser_water_temperature_difference` | K | Condenser water temperature difference. |
| `condenser_water_enthalpy_difference` | J/kg | Condenser water enthalpy difference. |
| `cooling_tower_set_reference_temperature_condenser_water` | °C | Cooling tower set reference temperature for the condenser water, i.e. the temperature at which condenser water leaves the cooling tower. |
| `cooling_tower_set_reference_temperature_wet_bulb` | °C | Cooling tower set reference temperature for the wet bulb ambient air temperature. |
| `cooling_tower_set_reference_temperature_slope` | °C | Cooling tower reference temperature slope, used to model the cooling tower efficiency. |
| `cooling_tower_set_ventilation_factor` | - | Cooling tower set ventilation factor, used to model the ventilation requirements depending on the condenser water flow. |

### `der_ev_chargers`

Supplementary flexible EV charger definition. Flexible EV chargers require additional time series / schedule definitions¹ as follows:

1. Maximum charging power time series: Maximum permitted charging power at the charging station. This should be determined based on the rated power of available chargers and the number of vehicles connected to chargers at each time step.
2. Maximum discharging power time series: Maximum permitted discharging power at the charging station. This should be determined based on the rated power of available chargers and the number of vehicles connected to chargers at each time step.
3. Maximum energy time series: Maximum value of aggregated charged energy across all vehicles. This should be determined based on the battery size or energy demand of all vehicles connected to chargers at each time step.
4. Departing energy time series: Aggregated energy demand of all departing vehicles. This should be determined based on the required energy of all vehicles which are departing from the charging station at each time step.

| Column | Unit | Description |
| --- |:---:| --- |
| `definition_name` | | Unique definition identifier. Corresponds to `definition_name` in `der_models`. |
| `maximum_charging_definition_type` | | Definition type selector for maximum charging power time series. Choices¹: `timeseries`, `schedule`. |
| `maximum_charging_definition_name` | | Definition identifier for maximum charging power time series, which corresponds to `definition_name` in either `der_timeseries`, `der_schedules`. |
| `maximum_discharging_definition_type` | | Definition type selector for maximum discharging power time series. Choices¹: `timeseries`, `schedule`. |
| `maximum_discharging_definition_name` | | Definition identifier for maximum discharging power time series, which corresponds to `definition_name` in either `der_timeseries`, `der_schedules`. |
| `maximum_energy_definition_type` | | Definition type selector for maximum energy time series. Choices¹: `timeseries`, `schedule`. |
| `maximum_energy_definition_name` | | Definition identifier for maximum energy time series, which corresponds to `definition_name` in either `der_timeseries`, `der_schedules`. |
| `departing_energy_definition_type` | | Definition type selector for departing energy time series. Choices¹: `timeseries`, `schedule`. |
| `departing_energy_definition_name` | | Definition identifier for departing energy time series, which corresponds to `definition_name` in either `der_timeseries`, `der_schedules`. |

¹ Time series values can be defined either directly as a time series in `der_timeseries` or as a schedule in `der_schedules`, where the latter describes recurring schedules based on weekday / time of day.
