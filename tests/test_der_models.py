"""Test DER models."""

import unittest

import fledge

logger = fledge.config.get_logger(__name__)


class TestDERModels(unittest.TestCase):

    # TODO: Add tests for new DER ode types.

    def test_fixed_load_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])

        # Get result.
        fledge.utils.log_time("test_fixed_load_model", log_level='info', logger_object=logger)
        fledge.der_models.FixedLoadModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'fixed_load', 'der_name'].iat[0]
        )
        fledge.utils.log_time("test_fixed_load_model", log_level='info', logger_object=logger)

    def test_fixed_ev_charger_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])

        # Get result.
        fledge.utils.log_time("test_fixed_ev_charger_model", log_level='info', logger_object=logger)
        fledge.der_models.FixedEVChargerModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'fixed_ev_charger', 'der_name'].iat[0]
        )
        fledge.utils.log_time("test_fixed_ev_charger_model", log_level='info', logger_object=logger)

    def test_fixed_generator_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])

        # Get result.
        fledge.utils.log_time("test_fixed_generator_model", log_level='info', logger_object=logger)
        fledge.der_models.FixedGeneratorModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'fixed_generator', 'der_name'].iat[0]
        )
        fledge.utils.log_time("test_fixed_generator_model", log_level='info', logger_object=logger)

    def test_flexible_load_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])

        # Get result.
        fledge.utils.log_time("test_flexible_load_model", log_level='info', logger_object=logger)
        fledge.der_models.FlexibleLoadModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'flexible_load', 'der_name'].iat[0]
        )
        fledge.utils.log_time("test_flexible_load_model", log_level='info', logger_object=logger)

    def test_flexible_generator_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])

        # Get result.
        fledge.utils.log_time("test_flexible_generator_model", log_level='info', logger_object=logger)
        fledge.der_models.FlexibleGeneratorModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'flexible_generator', 'der_name'].iat[0]
        )
        fledge.utils.log_time("test_flexible_generator_model", log_level='info', logger_object=logger)

    def test_storage_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])

        # Get result.
        fledge.utils.log_time("test_storage_model", log_level='info', logger_object=logger)
        fledge.der_models.StorageModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'storage', 'der_name'].iat[0]
        )
        fledge.utils.log_time("test_storage_model", log_level='info', logger_object=logger)

    def test_flexible_building_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])

        # Get result.
        fledge.utils.log_time("test_flexible_building_model", log_level='info', logger_object=logger)
        fledge.der_models.FlexibleBuildingModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'flexible_building', 'der_name'].iat[0]
        )
        fledge.utils.log_time("test_flexible_building_model", log_level='info', logger_object=logger)

    def test_cooling_plant_model(self):
        # Obtain test data.
        der_data = fledge.data_interface.DERData('paper_2021_troitzsch_dlmp_scenario_6_7_8')

        # Get result.
        fledge.utils.log_time("test_cooling_plant_model", log_level='info', logger_object=logger)
        fledge.der_models.CoolingPlantModel(
            der_data,
            # Take first DER of this type. Test will fail if no DERs of this type defined in the test scenario.
            der_data.ders.loc[der_data.ders.loc[:, 'der_type'] == 'cooling_plant', 'der_name'].iat[0]
        )
        fledge.utils.log_time("test_cooling_plant_model", log_level='info', logger_object=logger)

    def test_der_model_set(self):
        # Get result.
        fledge.utils.log_time("test_der_model_set", log_level='info', logger_object=logger)
        fledge.der_models.DERModelSet(fledge.config.config['tests']['scenario_name'])
        fledge.utils.log_time("test_der_model_set", log_level='info', logger_object=logger)


if __name__ == '__main__':
    unittest.main()
