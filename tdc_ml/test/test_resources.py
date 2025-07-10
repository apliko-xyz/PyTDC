import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import unittest
import shutil

from pandas import DataFrame


class TestResources(unittest.TestCase):
    pass


class TestPrimeKG(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    def test_node_retrieval(self):
        from tdc_ml.resource import PrimeKG
        data = PrimeKG(path='./data')
        drug_feature = data.get_features(feature_type='drug')
        data.to_nx()
        l = data.get_node_list('disease')
        assert "1" in l and "9997" in l


class TestPINNACLE(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    def test_mg_ppi_load(self):
        from tdc_ml.resource.pinnacle import PINNACLE
        pinnacle = PINNACLE()
        assert isinstance(pinnacle.get_ppi(), DataFrame)
        assert isinstance(pinnacle.get_mg(), DataFrame)
        assert len(pinnacle.get_ppi()) > 0
        assert len(pinnacle.get_mg()) > 0
        embeds = pinnacle.get_embeds()
        assert isinstance(embeds, DataFrame)
        assert len(embeds) > 0, "PINNACLE embeds is empty"

    def test_embeddings(self):
        from tdc_ml.resource.pinnacle import PINNACLE
        pinnacle = PINNACLE()
        embeds = pinnacle.get_embeds()
        assert isinstance(embeds, DataFrame)
        assert len(embeds) > 0, "PINNACLE embeds is empty"
        keys = pinnacle.get_keys()
        assert isinstance(keys, DataFrame)
        assert len(keys) > 0, "PINNACLE keys is empty"
        assert len(keys) == len(embeds), "{} vs {}".format(
            len(keys), len(embeds))
        num_targets = len(keys["target"].unique())
        num_cells = len(keys["cell type"].unique())
        all_entries = embeds.index
        prots = [x.split("--")[0] for x in all_entries]
        cells = [x.split("--")[1] for x in all_entries]
        assert len(
            set(prots)) == num_targets, "{} vs {} for target proteins".format(
                len(prots), num_targets)
        assert len(set(cells)) == num_cells, "{} vs {} for cell_types".format(
            len(cells), num_cells)

    def test_exp_data(self):
        from tdc_ml.resource.pinnacle import PINNACLE
        pinnacle = PINNACLE()
        exp_data = pinnacle.get_exp_data()
        assert isinstance(exp_data, DataFrame)
        assert len(exp_data) > 0, "PINNACLE exp_data is empty"

    def tearDown(self):
        try:
            print(os.getcwd())
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        except:
            pass


class TestPharmoneMap(unittest.TestCase):

    def setUp(self):
        print(os.getcwd())
        pass

    def test_get_data(self):
        from tdc_ml.resource.pharmone import PharmoneMap

        resource = PharmoneMap()
        data = resource.get_data()
        assert isinstance(data, DataFrame), type(data)
        assert "Compound" in data.columns
        assert "Target_ID" in data.columns
        assert "pXC50" in data.columns

    def tearDown(self):
        try:
            print(os.getcwd())
            shutil.rmtree(os.path.join(os.getcwd(), "data"))
        except:
            pass


if __name__ == "__main__":
    unittest.main()
