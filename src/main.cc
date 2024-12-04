#include <immintrin.h>
#include <iostream>
#include <vector>
#include <type_traits>

using namespace std;





template<typename T>
class Vector {
private:
    size_t simd_size = 256 / (sizeof(T) * 8);
    size_t simd_end  = simd_size - 1;
    std::vector<T> v_; 

public:
    static_assert(std::is_arithmetic<T>::value, "Vector can only be used with numerical types");

    Vector(int size, T def = 0): v_(size, def) {}
    Vector(std::initializer_list<T> _data): v_(_data) {}

    void push_back(T _data) {
        this->v_.push_back(_data);
    }

    size_t size() {
        return v_.size();
    }

    size_t capacity() {
        return v_.capacity();
    }

    T& operator[](size_t index) {
        if (index >= v_.size()) {
            throw std::out_of_range("Index out of bound");
        }
        return v_[index];
    }

    Vector<T> operator+(Vector<T>& other) {
        Vector<T> new_vec(this->v_.size());   
        size_t i = 0;

        if constexpr (std::is_same<T, double>::value) {

            for (; i + simd_end < v_.size(); i += simd_size) {

                __m256d veca = _mm256_loadu_pd(&v_[i]);
                __m256d vecb = _mm256_loadu_pd(&other[i]);


                __m256d vecres = _mm256_add_pd(veca, vecb);


                _mm256_storeu_pd(&new_vec[i], vecres);
            }
        } else if constexpr (std::is_same<T, float>::value) {

            for (; i + simd_end < v_.size(); i += simd_size) {

                __m256 veca = _mm256_loadu_ps(&v_[i]);
                __m256 vecb = _mm256_loadu_ps(&other[i]);

                __m256 vecres = _mm256_add_ps(veca, vecb);

                _mm256_storeu_ps(&new_vec[i], vecres);
            }
        } else if constexpr (std::is_same<T, int>::value) {

            for (; i + simd_end < v_.size(); i += simd_size) {
                __m256i veca = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&v_[i]));
                __m256i vecb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&other[i]));

                __m256i vecres = _mm256_add_epi32(veca, vecb);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&new_vec[i]), vecres);
            }
        } else if constexpr (std::is_same<T, long long>::value) {
            for (; i + simd_end < v_.size(); i += simd_size) {
                __m256i veca = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&v_[i]));
                __m256i vecb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&other[i]));

                __m256i vecres = _mm256_add_epi64(veca, vecb);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&new_vec[i]), vecres);
            }
        }

        for (; i < size(); i++) {
            new_vec[i] = this->v_[i] + other[i];
        }

        return new_vec;
    }

};

int main() {
    Vector<double> a(8, 1.4);
    Vector<double> b(8, 1.3);

    auto c = a + b;

    for(int i = 0;i < c.size();i++){
        cout << c[0] << " ";
    }
    cout << endl;

    return 0;
}
