import pandas as pd
import numpy as np
import gower
import ast
from collections import Counter
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from src.domain.models import ClusterEntity, ClusteringResult


class ClusteringService:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.feature_map = {
            "Зарплата": "salary_avg",
            "Минимальный опыт": "min_experience_years",
            "Название вакансии": "name",
            "График работы": "schedule_name",
            "Местность": "area_name"
        }

    def perform_clustering(self, selected_features: list, k_range: range) -> ClusteringResult:
        if not selected_features:
            raise ValueError("Не выбраны признаки")

        target_cols = [self.feature_map[f] for f in selected_features]
        num_cols = [c for c in target_cols if c in ['salary_avg', 'min_experience_years']]
        cat_cols = [c for c in target_cols if c not in num_cols]

        X = self.df[target_cols].dropna().copy()

        if X.empty:
            raise ValueError("Нет данных после удаления пропусков")

        X_proc = X.copy()

        if num_cols:
            scaler = StandardScaler()
            X_proc[num_cols] = scaler.fit_transform(X[num_cols])

        if cat_cols:
            X_proc[cat_cols] = X_proc[cat_cols].astype(str)

        if not cat_cols:
            return self._run_kmeans(X, X_proc, num_cols, k_range)
        elif not num_cols:
            return self._run_kmodes(X, X_proc, cat_cols, k_range)
        else:
            return self._run_kprototypes(X, X_proc, num_cols, cat_cols, k_range)

    def _run_kmeans(self, X_orig, X_proc, num_cols, k_range):
        best_k, best_score = self._find_best_k(
            k_range,
            lambda k: KMeans(n_clusters=k, n_init=3, random_state=42),
            X_proc,
            metric='euclidean'
        )
        model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        clusters = model.fit_predict(X_proc)
        return self._build_result(X_orig, clusters, best_k, best_score, "K-Means", num_cols, [])

    def _run_kmodes(self, X_orig, X_proc, cat_cols, k_range):
        best_k, best_score = self._find_best_k(
            k_range,
            lambda k: KModes(n_clusters=k, init='Cao', n_init=1, verbose=0),
            X_proc,
            metric='hamming'
        )
        model = KModes(n_clusters=best_k, init='Cao', n_init=5, verbose=0)
        clusters = model.fit_predict(X_proc)
        return self._build_result(X_orig, clusters, best_k, best_score, "K-Modes", [], cat_cols)

    def _run_kprototypes(self, X_orig, X_proc, num_cols, cat_cols, k_range):
        cat_indices = [X_proc.columns.get_loc(c) for c in cat_cols]
        dist_matrix = gower.gower_matrix(X_proc)

        best_k, best_score = self._find_best_k(
            k_range,
            lambda k: KPrototypes(n_clusters=k, init='Cao', n_init=1, verbose=0, random_state=42),
            X_proc.values,
            metric='precomputed',
            X_score=dist_matrix,
            fit_params={'categorical': cat_indices}
        )

        final_model = KPrototypes(n_clusters=best_k, init='Cao', n_init=3, verbose=0, random_state=42)
        clusters = final_model.fit_predict(X_proc.values, categorical=cat_indices)

        return self._build_result(X_orig, clusters, best_k, best_score, "K-Prototypes", num_cols, cat_cols)

    def _find_best_k(self, k_range, model_factory, X_train, metric, X_score=None, fit_params=None):
        best_score = -1
        best_k = k_range.start
        if X_score is None: X_score = X_train
        if fit_params is None: fit_params = {}

        for k in k_range:
            try:
                model = model_factory(k)
                labels = model.fit_predict(X_train, **fit_params)
                if len(set(labels)) < 2 or len(set(labels)) >= len(labels): continue
                score = silhouette_score(X_score, labels, metric=metric)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue
        return best_k, max(best_score, 0)


    def _build_result(self, X_orig, clusters, k, score, method_name, num_cols, cat_cols):
        df_result = self.df.loc[X_orig.index].copy()
        df_result['cluster'] = clusters

        all_salaries = df_result['salary_avg'].dropna()
        q30 = all_salaries.quantile(0.3) if not all_salaries.empty else 0
        q70 = all_salaries.quantile(0.7) if not all_salaries.empty else 0

        cluster_entities = []
        for i in range(k):
            cluster_data = df_result[df_result['cluster'] == i]
            title_parts = []
            if 'name' in cluster_data.columns:
                total_count = len(cluster_data)
                for name, count in cluster_data['name'].value_counts().items():
                    if (count / total_count) >= 0.30:
                        title_parts.append(name)

            title_base = " / ".join(title_parts) if title_parts else f"Группа #{i + 1}"

            cluster_median = cluster_data['salary_avg'].median()
            salary_tag = ""
            if pd.notnull(cluster_median) and q70 > 0:
                if cluster_median > q70:
                    salary_tag = "🤑 Высокая ЗП"
                elif cluster_median > q30:
                    salary_tag = "💵 Нормальная ЗП"
                else:
                    salary_tag = "📉 Маленькая ЗП"

            final_title = f"{title_base} ({salary_tag})" if salary_tag else title_base
            top_skills = []
            if 'skills_list' in cluster_data.columns:
                all_skills = []
                for val in cluster_data['skills_list'].dropna():
                    try:
                        if isinstance(val, str):
                            parsed = ast.literal_eval(val)
                            if isinstance(parsed, list):
                                all_skills.extend(parsed)
                        elif isinstance(val, list):
                            all_skills.extend(val)
                    except:
                        continue
                if all_skills:
                    top_skills = [skill for skill, count in Counter(all_skills).most_common(5)]
            remote_percentage = 0.0
            if 'schedule_name' in cluster_data.columns:
                total_vacs = len(cluster_data)
                if total_vacs > 0:
                    remote_count = cluster_data['schedule_name'].astype(str).str.contains('удален', case=False).sum()
                    remote_percentage = (remote_count / total_vacs) * 100
            avg_sal = cluster_data['salary_avg'].mean()
            avg_sal_str = f"{int(avg_sal):,} ₽".replace(",", " ") if pd.notnull(avg_sal) else "Не указана"

            desc_parts = []
            for col in cat_cols:
                if not cluster_data[col].mode().empty:
                    desc_parts.append(str(cluster_data[col].mode()[0]))
            for col in num_cols:
                mean_val = cluster_data[col].mean()
                if col == 'min_experience_years':
                    desc_parts.append(f"Опыт ~{mean_val:.1f} г.")

            description = ", ".join(desc_parts) if desc_parts else "Смешанная группа"

            entity = ClusterEntity(
                id=i + 1,
                title=final_title,
                description=description,
                vacancies_count=len(cluster_data),
                avg_salary=avg_sal_str,
                skills=top_skills,
                remote_rate=remote_percentage
            )
            cluster_entities.append(entity)

        return ClusteringResult(
            method_name=method_name,
            n_clusters=k,
            silhouette_score=score,
            clusters=cluster_entities
        )