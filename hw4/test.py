import cmath

def dft(data):
    N = len(data)
    result = [sum(data[n] * cmath.exp(-2j * cmath.pi * k * n / N) for n in range(N)) for k in range(N)]
    return result

def idft(coefficients):
    N = len(coefficients)
    result = [sum(coefficients[k] * cmath.exp(2j * cmath.pi * k * n / N) for k in range(N)) / N for n in range(N)]
    return result

# 資料
data = [5, 3, 7, 2, 6, 4]

# 計算DFT
fourier_coefficients = dft(data)

# 計算IDFT（可選，用於確認結果）
reconstructed_data = idft(fourier_coefficients)

# 輸出結果
print("傅立葉級數係數：", fourier_coefficients)
print("重構的資料（IDFT）：", [round(x.real) for x in reconstructed_data])