from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import math

from pandas_tasks import analyze_titanic
from correlation_tasks import analyze_brain_correlations
from grader_contracts.pandas_tasks import TitanicInput
from grader_contracts.correlation_tasks import BrainDataInput


class AnalysisTests(unittest.TestCase):
    def test_titanic_missing_values_boundary_and_duplicate_fares(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'titanic.csv'
            path.write_text('Pclass,Age,Survived,Fare\n1,30,1,100\n1,,0,100\n2,40,1,30\n2,20,0,20\n3,31,0,10\n3,10,1,5\n')
            result = analyze_titanic(TitanicInput(str(path)))
        self.assertEqual(result.row_count, 6)
        self.assertEqual(result.missing_by_column, {'Pclass': 0, 'Age': 1, 'Survived': 0, 'Fare': 0})
        self.assertEqual(result.adults_over_30_count, 2)
        self.assertEqual(result.mean_age_by_pclass, {1: 30., 2: 30., 3: 20.5})
        self.assertEqual(result.survival_rate_by_pclass, {1: .5, 2: .5, 3: .5})
        self.assertTrue(all(type(key) is int for key in result.mean_age_by_pclass))
        self.assertEqual(result.highest_fares, [100., 100., 30., 20., 10.])

    def test_brain_pairwise_missing_and_absolute_strength(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'brain.tsv'
            path.write_text(
                'Gender\tFSIQ\tVIQ\tPIQ\tWeight\tHeight\tMRI_Count\n'
                'Female\t4\t1\t1\tNA\t1\t1\n'
                'Female\t3\t3\t4\t3\t3\t2\n'
                'Female\t2\t2\t2\t2\t2\t3\n'
                'Female\t1\t4\t3\t4\t4\t4\n'
                'Male\t1\t1\t1\t1\t1\t1\n'
                'Male\t3\t3\t3\t3\t3\t2\n'
                'Male\t2\t2\t2\t2\t2\t3\n'
                'Male\t4\t4\t4\t4\t4\t4\n'
            )
            result = analyze_brain_correlations(BrainDataInput(str(path)))
        self.assertEqual((result.men_count, result.women_count), (4, 4))
        self.assertEqual(result.strongest_mri_feature, 'FSIQ')
        self.assertAlmostEqual(result.women_mri_correlation['FSIQ'], -1.)
        self.assertAlmostEqual(result.women_mri_correlation['Weight'], .5)
        self.assertAlmostEqual(result.women_mri_correlation['VIQ'], .8)
        self.assertAlmostEqual(result.men_mri_correlation['FSIQ'], .8)
        self.assertNotIn('MRI_Count', result.men_mri_correlation)

    def test_original_datasets(self):
        root = Path(__file__).resolve().parent
        titanic = analyze_titanic(TitanicInput(str(root / 'titanic.csv')))
        brain = analyze_brain_correlations(BrainDataInput(str(root / 'brainsize.txt')))
        self.assertEqual(titanic.row_count, 891)
        self.assertEqual(titanic.missing_by_column['Age'], 177)
        self.assertEqual((brain.men_count, brain.women_count), (20, 20))
        self.assertEqual(len(brain.men_mri_correlation), 5)
        self.assertTrue(all(math.isfinite(v) for v in brain.men_mri_correlation.values()))


if __name__ == '__main__':
    unittest.main()
