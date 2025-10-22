import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.font_manager as fm

# ==== 日本語フォント設定（Colab・Streamlit共通）====
try:
    font_path = "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
except Exception:
    plt.rcParams['font.family'] = "Arial"

plt.rcParams['axes.unicode_minus'] = False

# ==== グラフのスタイル共通設定 ====
sns.set(style="whitegrid", palette="pastel")

def show_distribution(df, column, bins=30, title_ja=None):
    """
    指定したカラムの分布をヒストグラムで表示
    """
    if column not in df.columns:
        print(f"⚠️ カラム {column} が存在しません。スキップします。")
        return

    plt.figure(figsize=(6, 4))
    sns.histplot(df[column].dropna(), bins=bins, kde=True, color="#85C1E9")
    plt.title(title_ja if title_ja else f"{column} の分布")
    plt.xlabel(title_ja if title_ja else column)
    plt.ylabel("人数")
    plt.tight_layout()
    plt.show()


def show_boxplot(df, columns, title_ja="選択した特徴量の箱ひげ図"):
    """
    複数カラムをまとめて箱ひげ図で表示
    """
    valid_columns = [c for c in columns if c in df.columns]
    if not valid_columns:
        print("⚠️ 有効なカラムがありません。スキップします。")
        return

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df[valid_columns], palette="coolwarm")
    plt.title(title_ja)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def visualize_overall(result_dict, title_ja="各特徴量のスコア比較"):
    """
    calculate_metrics の結果辞書を棒グラフで表示
    """
    if not result_dict:
        print("⚠️ 結果が空です。スキップします。")
        return

    plt.figure(figsize=(6, 4))
    names = list(result_dict.keys())
    values = list(result_dict.values())
    sns.barplot(x=names, y=values, palette="cool")
    plt.title(title_ja)
    plt.ylabel("スコア")
    plt.ylim(0, max(values) * 1.2 if len(values) else 1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
