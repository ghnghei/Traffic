#报表生成
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os


class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_csv_report(self, count_data):
        """生成CSV格式报表"""
        df = pd.DataFrame(count_data).T.fillna(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"{self.output_dir}/traffic_report_{timestamp}.csv"
        df.to_csv(csv_path)
        return csv_path

    def generate_visual_report(self, count_data, directions=None):
        """生成可视化报表"""
        # 车辆类型计数图
        df = pd.DataFrame(count_data).T.fillna(0)
        ax = df.sum(axis=1).plot(kind='bar', title='Vehicle Count by Type')
        plt.ylabel('Count')
        plt.tight_layout()

        count_path = f"{self.output_dir}/counts_{datetime.now().strftime('%H%M%S')}.png"
        plt.savefig(count_path)
        plt.close()

        # 方向流量图
        if directions:
            dir_df = pd.DataFrame(directions)
            ax = dir_df.plot(kind='bar', stacked=True, title='Traffic Direction Flow')
            plt.ylabel('Count')
            plt.tight_layout()

            dir_path = f"{self.output_dir}/directions_{datetime.now().strftime('%H%M%S')}.png"
            plt.savefig(dir_path)
            plt.close()

            return count_path, dir_path
        return count_path, None

    def generate_summary(self, total_counts, direction_counts=None):
        """生成汇总报告"""
        report = f"Traffic Analysis Report\nDate: {datetime.now()}\n\n"
        report += "Total Vehicle Counts:\n"

        for vehicle, count in total_counts.items():
            report += f"- {vehicle.title()}: {count}\n"

        if direction_counts:
            report += "\nDirectional Flow:\n"
            for vehicle, directions in direction_counts.items():
                for direction, count in directions.items():
                    report += f"- {vehicle.title()} moving {direction}: {count}\n"

        return report