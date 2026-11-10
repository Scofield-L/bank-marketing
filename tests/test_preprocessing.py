"""数据接口回归测试：划分、防泄漏、未知类别和训练折采样。"""
import sys
from pathlib import Path
import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from bank_preprocessing import load_data, split_data, make_preprocessor, make_model_pipeline


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_data()

    def test_split_reproducible_and_aligned(self):
        a, b, ya, yb = split_data(self.frame)
        a2, b2, _, _ = split_data(self.frame)
        self.assertTrue(a.index.equals(a2.index))
        self.assertTrue(b.index.equals(b2.index))
        self.assertFalse(set(a.index) & set(b.index))
        self.assertEqual(set(a.index) | set(b.index), set(self.frame.index))
        self.assertTrue(a.index.equals(ya.index))
        self.assertTrue(b.index.equals(yb.index))
        self.assertNotIn('duration', a)
        self.assertNotIn('y', a)
        self.assertLess(abs(ya.mean() - yb.mean()), 0.001)

    def test_train_only_statistics_and_unseen_category(self):
        a, b, _, _ = split_data(self.frame)
        p = make_preprocessor()
        p.fit(a)
        means = p.named_transformers_['numeric'].named_steps['scale'].mean_
        columns = p.transformers_[0][2]
        np.testing.assert_allclose(means, a[columns].mean().to_numpy())
        altered = b.iloc[:2].copy()
        altered['job'] = 'unseen_job'
        altered['age'] = np.nan
        self.assertTrue(np.isfinite(p.transform(altered)).all())
        np.testing.assert_array_equal(means, p.named_transformers_['numeric'].named_steps['scale'].mean_)

    def test_all_sampling_pipelines_cross_validation(self):
        # 固定的小型训练子集，测试不读取测试集、不预先拟合全量数据。
        a, _, ya, _ = split_data(self.frame)
        ids = ya.groupby(ya).head(120).index
        for strategy in ['none', 'class_weight', 'undersample', 'smote']:
            with self.subTest(strategy=strategy):
                pipeline = make_model_pipeline(LogisticRegression(max_iter=1000), strategy)
                scores = cross_val_score(pipeline, a.loc[ids], ya.loc[ids],
                                         cv=StratifiedKFold(2, shuffle=True, random_state=42),
                                         scoring='roc_auc', error_score='raise')
                self.assertTrue(np.isfinite(scores).all())


if __name__ == '__main__':
    unittest.main()
