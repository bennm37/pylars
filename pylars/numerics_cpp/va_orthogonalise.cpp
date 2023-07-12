#include <iostream>
#include <complex>
#include <vector>
#include <cmath>

typedef std::complex<double> Complex;

std::vector<std::vector<Complex> > va_orthogonalise(const std::vector<Complex>& Z, int n) {
    // if (Z.size() != 1) {
    //     throw std::invalid_argument("Z must be a column vector");
    // }
    
    int m = Z.size();
    Complex m_complex = Complex(m, 0.0);
    std::vector<std::vector<Complex> > H(n + 1, std::vector<Complex>(n));
    std::vector<std::vector<Complex> > Q(m, std::vector<Complex>(n + 1));
    std::vector<Complex> q(m, 1.0);
    
    for (int i = 0; i < m; ++i) {
        Q[i][0] = q[i];
    }
    
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < m; ++i) {
            q[i] = Z[0] * Q[i][k];
        }
        
        for (int j = 0; j <= k; ++j) {
            Complex dotProduct = Complex(0.0, 0.0);
            for (int i = 0; i < m; ++i) {
                dotProduct += std::conj(Q[i][j]) * q[i];
            }
            H[j][k] = dotProduct / m_complex;
            
            for (int i = 0; i < m; ++i) {
                q[i] -= H[j][k] * Q[i][j];
            }
        }
        
        double norm = 0.0;
        for (int i = 0; i < m; ++i) {
            norm += std::norm(q[i]);
        }
        H[k + 1][k] = std::sqrt(norm) / std::sqrt(m);
        
        for (int i = 0; i < m; ++i) {
            Q[i][k + 1] = q[i] / H[k + 1][k];
        }
    }
    
    // std::vector<std::vector<Complex> > hessenbergs;
    // hessenbergs.emplace_back(H);
    
    return H;
}

int main() {
    std::vector<Complex> Z = {{1.0, 0.0}, {2.0, 0.0}, {1.0, 1.0}};
    int n = 2;
    
    std::vector<std::vector<Complex> > hessenbergs = va_orthogonalise(Z, n);
    
    std::cout << "Hessenberg matrix:" << std::endl;
    // print the hessenberg matrix 
    for (int i = 0; i < hessenbergs.size(); ++i) {
        for (int j = 0; j < hessenbergs[i].size(); ++j) {
            std::cout << hessenbergs[i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}
