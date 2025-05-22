import os
import json
import pandas as pd
from pathlib import Path

def extract_nerf_results(base_folder="./output/dnerf", output_csv="nerf_results.csv"):
    """
    ./output/dnerf 폴더 내의 각 데이터셋 폴더에서 results.json을 읽어
    PSNR, SSIM, LPIPS-alex 값을 추출하여 CSV로 저장
    """
    
    results = []
    base_path = Path(base_folder)
    
    # 기본 폴더가 존재하는지 확인
    if not base_path.exists():
        print(f"Error: 폴더 '{base_folder}'가 존재하지 않습니다.")
        return
    
    # 각 데이터셋 폴더 순회
    for dataset_folder in base_path.iterdir():
        if dataset_folder.is_dir():
            results_file = dataset_folder / "results.json"
            
            # results.json 파일이 존재하는지 확인
            if results_file.exists():
                try:
                    # JSON 파일 읽기
                    with open(results_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 첫 번째 키의 데이터 추출 (보통 "ours_20000" 같은 형태)
                    if data:
                        first_key = list(data.keys())[0]
                        metrics = data[first_key]
                        
                        # 필요한 메트릭 추출
                        result_row = {
                            'dataset': dataset_folder.name,
                            'method': first_key,
                            'PSNR': metrics.get('PSNR', None),
                            'SSIM': metrics.get('SSIM', None),
                            'LPIPS-alex': metrics.get('LPIPS-alex', None)
                        }
                        
                        results.append(result_row)
                        print(f"✓ {dataset_folder.name}: PSNR={result_row['PSNR']:.4f}, SSIM={result_row['SSIM']:.4f}, LPIPS-alex={result_row['LPIPS-alex']:.6f}")
                    
                except json.JSONDecodeError as e:
                    print(f"⚠ JSON 파싱 오류 ({dataset_folder.name}): {e}")
                except Exception as e:
                    print(f"⚠ 파일 읽기 오류 ({dataset_folder.name}): {e}")
            else:
                print(f"⚠ results.json 파일이 없음: {dataset_folder.name}")
    
    # 결과가 있는 경우 CSV로 저장
    if results:
        df = pd.DataFrame(results)
        
        # PSNR 기준으로 내림차순 정렬 (높을수록 좋음)
        df = df.sort_values('PSNR', ascending=False)
        
        # CSV 저장
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n📊 결과가 '{output_csv}' 파일로 저장되었습니다.")
        print(f"총 {len(results)}개 데이터셋의 결과를 처리했습니다.")
        
        # 간단한 통계 출력
        print("\n📈 통계 요약:")
        print(f"PSNR - 평균: {df['PSNR'].mean():.4f}, 최고: {df['PSNR'].max():.4f}, 최저: {df['PSNR'].min():.4f}")
        print(f"SSIM - 평균: {df['SSIM'].mean():.4f}, 최고: {df['SSIM'].max():.4f}, 최저: {df['SSIM'].min():.4f}")
        print(f"LPIPS-alex - 평균: {df['LPIPS-alex'].mean():.6f}, 최고: {df['LPIPS-alex'].min():.6f}, 최저: {df['LPIPS-alex'].max():.6f}")
        
        return df
    else:
        print("⚠ 처리할 데이터가 없습니다.")
        return None

def extract_all_methods_results(base_folder="./output/dnerf", output_csv="nerf_results_all_methods.csv"):
    """
    results.json에 여러 메소드가 있는 경우 모든 메소드를 추출하는 버전
    """
    
    results = []
    base_path = Path(base_folder)
    
    if not base_path.exists():
        print(f"Error: 폴더 '{base_folder}'가 존재하지 않습니다.")
        return
    
    for dataset_folder in base_path.iterdir():
        if dataset_folder.is_dir():
            results_file = dataset_folder / "results.json"
            
            if results_file.exists():
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 모든 메소드에 대해 처리
                    for method_name, metrics in data.items():
                        result_row = {
                            'dataset': dataset_folder.name,
                            'method': method_name,
                            'PSNR': metrics.get('PSNR', None),
                            'SSIM': metrics.get('SSIM', None),
                            'LPIPS-alex': metrics.get('LPIPS-alex', None)
                        }
                        results.append(result_row)
                        
                except Exception as e:
                    print(f"⚠ 오류 ({dataset_folder.name}): {e}")
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(['dataset', 'PSNR'], ascending=[True, False])
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"📊 모든 메소드 결과가 '{output_csv}' 파일로 저장되었습니다.")
        return df
    
    return None

if __name__ == "__main__":
    # 기본 실행: 각 데이터셋의 첫 번째 메소드만 추출
    print("🔍 NeRF 결과 추출을 시작합니다...")
    df = extract_nerf_results(base_folder="./output/sds/dnerf_10000", output_csv="dnerf_10000sds_results.csv")
    
    # 모든 메소드를 추출하고 싶다면 아래 주석을 해제하세요
    # print("\n🔍 모든 메소드 결과 추출...")
    # df_all = extract_all_methods_results()