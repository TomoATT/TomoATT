#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <math.h>
#include <string.h>
#include <fstream>
#include <sys/stat.h>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <new> // for std::bad_alloc
#include <cstdlib> // for std::exit


// chec if compiler is gcc 7.5 or older
//#if __GNUC__ == 7 && __GNUC_MINOR__ <= 5
//#include <experimental/filesystem>
//#define GNUC_7_5
//#elif __cplusplus > 201402L // compilers supporting c++17
//#include <filesystem>
//#endif

#include "config.h"


inline int mkpath(std::string s,mode_t mode) {
    size_t pos=0;
    std::string dir;
    int mdret = 0;

    if(s[s.size()-1]!='/'){
        // force trailing / so we can handle everything in loop
        s+='/';
    }

    while((pos=s.find_first_of('/',pos))!=std::string::npos){
        dir=s.substr(0,pos++);
        if(dir.size()==0) continue; // if leading / first time is 0 length
        if((mdret=mkdir(dir.c_str(),mode)) && errno!=EEXIST){
            return mdret;
        }
    }
    return mdret;
}



inline void create_output_dir(std::string dir_path){

    // create output directory if not exists (directories tree)
    if (world_rank == 0) {

//#if __cplusplus > 201402L && !defined(GNUC_7_5)
//        // this function requires c++17
//        if (!std::filesystem::exists(dir_path)){
//            std::filesystem::create_directories(dir_path);
//        } else {
//            std::cout << "Output directory already exists. Overwriting..." << std::endl;
//        }
//#else // compilers not supporting std++17
        // check if directory exists
        struct stat info;
        if (stat(dir_path.c_str(), &info) != 0){
            // directory does not exist
            // create directory
            int status = mkpath(dir_path, 0755);
            if (status != 0){
                std::cout << "Error: cannot create output directory" << std::endl;
                std::cout << "Exiting..." << std::endl;
                exit(1);
            }
        } else {
            // directory exists
            std::cout << "Output directory already exists. Overwriting..." << std::endl;
        }
//#endif

    }

}


inline bool is_file_exist(const char* fileName)
{
    return static_cast<bool>(std::ifstream(fileName));
}


inline void stdout_by_main(char const* str){
    if (sim_rank == 0 && inter_sub_rank == 0 && sub_rank == 0 && id_sim == 0)
        std::cout << str << std::endl;
}


inline void stdout_by_rank_zero(char const* str){
    if(world_rank == 0)
        std::cout << str << std::endl;
}


inline void parse_options(int argc, char* argv[]){
    bool input_file_found = false;

    for (int i = 1; i < argc; i++){
        if(strcmp(argv[i], "-v") == 0)
            if_verbose = true;
        else if (strcmp(argv[i],"-i") == 0){
            input_file = argv[i+1];
            input_file_found = true;
        }
        else if (strcmp(argv[i], "-h") == 0){
            stdout_by_main("usage: mpirun -np 4 TOMOATT -i input_params.yaml");
            exit(EXIT_SUCCESS);
        }
    }

    // error if input_file is not found
    if(!input_file_found){
        stdout_by_main("usage: mpirun -np 4 TOMOATT -i input_params.yaml");
        std::cout << "Error: input parameter file not found" << std::endl;
        exit(EXIT_FAILURE);
    }
}


inline void parse_options_srcrec_weight(int argc, char* argv[]){
    bool input_file_found = false;

    for (int i = 1; i < argc; i++){
        if(strcmp(argv[i], "-v") == 0)
            if_verbose = true;
        else if (strcmp(argv[i],"-i") == 0){
            input_file = argv[i+1];
            input_file_found = true;
        } else if (strcmp(argv[i],"-r") == 0){
            // reference value
            ref_value = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-o") == 0){
            // get output filename
            output_file_weight = argv[i+1];
        }
    }

    // error if input_file is not found
    if(!input_file_found){
        stdout_by_main("usage: ./SrcRecWeight -i srcrec_file.txt -r 10.0 -o srcrec_weight.txt");
        std::cout << argc << std::endl;
        exit(EXIT_FAILURE);
    }
}


template <typename T>
inline int check_data_type(T const& data){
    if (std::is_same<T,bool>::value){
        return 0;
    } else if (std::is_same<T, int>::value){
        return 1;
    } else if (std::is_same<T, float>::value){
        return 2;
    } else if (std::is_same<T, double>::value){
        return 3;
    } else {
        std::cout << "Error: custom real type is not float or double" << std::endl;
        std::cout << "Please check the definition of CUSTOMREAL in io.h" << std::endl;
        std::cout << "Exiting..." << std::endl;
        exit(1);
    }
}


template <typename T>
inline T my_square(T const& a){
    return a * a;
}


// defined function is more than 2 times slower than inline function
//#define my_square(a) (a * a)


template <typename T>
T* allocateMemory(size_t size, int allocationID) {
    try {
        T* ptr = new T[size];
        return ptr;
    } catch (const std::bad_alloc& e) {
        // Handle memory allocation failure
        std::cerr << "Memory allocation failed for ID " << allocationID << ": " << e.what() << std::endl;
        std::exit(EXIT_FAILURE); // Exit the program on allocation failure
    }
}


inline void RLonLat2xyz(CUSTOMREAL lon, CUSTOMREAL lat, CUSTOMREAL R, CUSTOMREAL& x, CUSTOMREAL& y, CUSTOMREAL& z){
    /*
    lon : longitude in radian
    lat : latitude in radian
    R : radius

    x : x coordinate in m
    y : y coordinate in m
    z : z coordinate in m
    */
    //x = R*cos(lat*DEG2RAD)*cos(lon*DEG2RAD);
    //y = R*cos(lat*DEG2RAD)*sin(lon*DEG2RAD);
    //z = R*sin(lat*DEG2RAD);
    x = R*cos(lat)*cos(lon);
    y = R*cos(lat)*sin(lon);
    z = R*sin(lat);

}


// calculate epicentral distance in radian from lon lat in radian
inline void Epicentral_distance_sphere(const CUSTOMREAL lat1, const CUSTOMREAL lon1, \
                                       const CUSTOMREAL lat2, const CUSTOMREAL lon2, \
                                       CUSTOMREAL& dist) {

    // calculate epicentral distance in radian
    //dist = std::abs(acos(sin(lat1)*sin(lat2)+cos(lat1)*cos(lat2)*cos(lon2-lon1)));
    CUSTOMREAL lon_dif = lon2 - lon1;

    dist = std::atan2(std::sqrt((my_square(std::cos(lat2) * std::sin(lon_dif)) + \
                                 my_square(std::cos(lat1) * std::sin(lat2) - std::sin(lat1) * std::cos(lat2) * std::cos(lon_dif)))), \
                                 std::sin(lat1) * std::sin(lat2) + std::cos(lat1) * std::cos(lat2) * std::cos(lon_dif));

}




//calculate azimuth
inline void Azimuth_sphere(CUSTOMREAL lat1, CUSTOMREAL lon1, \
                             CUSTOMREAL lat2, CUSTOMREAL lon2, \
                             CUSTOMREAL& azi) {

    if (isZero(my_square((lat1-lat2)) \
        &&   + my_square((lon1-lon2)))){
        azi = _0_CR;
    } else {
        azi = atan2(sin(lon2-lon1)*cos(lat2), \
                    cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(lon2-lon1));
        if (azi < _0_CR)
            azi += _2_CR * PI;
    }
}


inline void WGS84ToCartesian(CUSTOMREAL& lon, CUSTOMREAL& lat, CUSTOMREAL R, \
                             CUSTOMREAL& x, CUSTOMREAL& y, CUSTOMREAL& z, \
                             CUSTOMREAL& lon_center, CUSTOMREAL& lat_center){

    // equatorial radius WGS84 major axis
    const static CUSTOMREAL equRadius = R_earth; // in m in this function
    const static CUSTOMREAL flattening = 1.0 / 298.257222101;

    const static CUSTOMREAL sqrEccentricity = flattening * (2.0 - flattening);

    const CUSTOMREAL lat_rad = lat - lat_center; // input should be already in radian
    const CUSTOMREAL lon_rad = lon - lon_center;
    const CUSTOMREAL alt = R - equRadius; // altitude (height above sea level)

    const CUSTOMREAL sinLat = sin(lat_rad);
    const CUSTOMREAL cosLat = cos(lat_rad);
    const CUSTOMREAL sinLon = sin(lon_rad);
    const CUSTOMREAL cosLon = cos(lon_rad);

    // Normalized radius
    const CUSTOMREAL normRadius = equRadius / sqrt(1.0 - sqrEccentricity * sinLat * sinLat);

    x = (normRadius + alt) * cosLat * cosLon;
    y = (normRadius + alt) * cosLat * sinLon;
    z = (normRadius * (1.0 - sqrEccentricity) + alt) * sinLat;

}


template <typename T>
inline T dot_product(T const* const a, T const* const b, int const& n){
    CUSTOMREAL result = _0_CR;
    for (int i = 0; i < n; i++){
        result += a[i] * b[i];
    }
    return result;
}

template <typename T>
inline T find_max(T const* const a, int const& n){
    T max = a[0];
    for (int i = 1; i < n; i++){
        if (a[i] > max){
            max = a[i];
        }
    }
    return max;
}

template <typename T>
inline T find_absmax(T const* const a, int const& n){
    T max = fabs(a[0]);
    for (int i = 1; i < n; i++){
        if (fabs(a[i]) > max){
            max = fabs(a[i]);
        }
    }
    return max;
}

template <typename T>
inline T calc_l2norm(T const* const a, int const& n){
    T result = _0_CR;
    for (int i = 0; i < n; i++){
        result += a[i] * a[i];
    }
    return result;
}


inline std::string int2string_zero_fill(int i) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(4) << i;
    return ss.str();
}


inline bool in_between(CUSTOMREAL const& a, CUSTOMREAL const& b, CUSTOMREAL const& c){
    // check if a is between b and c
    if ((a-b)*(a-c) <= _0_CR){
        return true;
    } else {
        return false;
    }
}


inline CUSTOMREAL calc_ratio_between(CUSTOMREAL const& a, CUSTOMREAL const& b, CUSTOMREAL const& c){
    // calculate ratio of a between b and c
    //if (b < c)
        return (a - b) / (c - b);
    //else
    //    return (a - b) / (b - c);
}


inline void linear_interpolation_1d_sorted(const CUSTOMREAL* const x1, const CUSTOMREAL* const y1, const int& n1, const CUSTOMREAL* x2, CUSTOMREAL* y2, const int& n2){
    // linear interpolation for sorted 1d array (monotonely increasing)
    // x1 : positions of the first array
    // y1 : values of the first array
    // n1 : size of the first array
    // x2 : positions of the second array
    // y2 : values of the second array
    // n2 : size of the second array
    int start = 0;
    for(int pt2=0; pt2<n2; pt2++){              // x2[pt2] <= x1[0]
        if (x2[pt2] <= x1[0]){
            y2[pt2] = y1[0];
        } else if (x2[pt2] >= x1[n1-1]){    // x2[pt2] >= x1[n1-1]
            y2[pt2] = y1[n1-1];
        } else {                            // x1[0] < x2[pt2] < x1[n1-1]
            for(int pt1=start; pt1<n1-1; pt1++){
                if (x2[pt2] >= x1[pt1] && x2[pt2] <= x1[pt1+1]){
                    y2[pt2] = y1[pt1] + (y1[pt1+1] - y1[pt1]) * (x2[pt2] - x1[pt1]) / (x1[pt1+1] - x1[pt1]);
                    start = pt1;
                    break;
                }
            }
        }
    }
}

template<typename T>
std::vector<CUSTOMREAL> linspace(T start_in, T end_in, int num_in)
{

  std::vector<CUSTOMREAL> linspaced;

  double start = static_cast<CUSTOMREAL>(start_in);
  double end = static_cast<CUSTOMREAL>(end_in);
  double num = static_cast<CUSTOMREAL>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}

#endif // UTILS_H