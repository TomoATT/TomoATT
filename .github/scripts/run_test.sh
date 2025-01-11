#!/bin/bash
#
# runs a test example case
#

# getting updated environment
if [ -f $HOME/.tmprc ]; then
    source $HOME/.tmprc
fi

WORKDIR=`pwd`
predir=${PREDIR}
dir=${TESTDIR}

# get number of sweep parallelization if exists, or 1 by default
nproc_sweep=${NPROC_SWEEP:-1}
# get number of processes for domain decomposition if exists, or 1 by default
nproc_dd=${NPROC_DD:-1}

# set number of threads for OpenMP to be equal to the number of NPROC_SWEEP
export OMP_NUM_THREADS=$nproc_sweep


# print info
echo "Running test case in a work directory: $WORKDIR"
echo `date`
echo
echo "******************************************************************"
echo
echo "Data directory: $predir"
echo "test directory: $dir"
echo
echo "******************************************************************"
echo


# test script for checking the result
my_test_inversion_small(){
    echo "Checking the result..."
    ln -s ../../utils/compare_models.py
    python compare_models.py -t ./test_model_true.h5 -r ./OUTPUT_FILES/final_model.h5
    if [[ $? -ne 0 ]]; then
        echo "Test failed!"
        exit 1
    fi
    echo "Test passed!"
}

#my_test_forward(){
#    echo "Checking the result..."
#    python compare_src_rec.py
#    if [[ $? -ne 0 ]]; then
#        echo "Test failed!"
#        exit 1
#    fi
#    echo "Test passed!"
#}
#
#my_test_forward_2(){
#    echo "Checking the result..."
#    python compare_src_rec_obj.py
#    if [[ $? -ne 0 ]]; then
#        echo "Test failed!"
#        exit 1
#    fi
#    echo "Test passed!"
#}



# test example
cd $dir

echo "Running test case in a work directory: $WORKDIR"

# run the test
sh run_this_example.sh $nproc_sweep $nproc_dd

# check the exit code
if [[ $? -ne 0 ]]; then
    echo "Test failed!"
    exit 1
fi

# smulation finished, check the result
echo
echo "simulation finished: `pwd`"
echo `date`
echo

# check the result
if [ "$TESTDIR" == "test/inversion_small/" ]; then
    my_test_inversion_small
    # clean up
    rm -rf OUTPUT_FILES* src_rec_test.dat time.txt *h5
# elif [ "$TESTDIR" == "examples/1_forward_accuracy_1st_upwind_isotropic/" ]; then
#     my_test_forward
#     # clean up
#     rm -rf OUTPUT_FILES* src_rec_true.dat time.txt *h5 models
# elif [ "$TESTDIR" == "examples/2_src_rec_data_read/" ]; then
#     # clean up
#     rm -rf OUTPUT_FILES* src_rec_true.dat time.txt *h5 models
# elif [ "$TESTDIR" == "examples/3_forward_accuracy_1st_cuthill_isotropic/" ]; then
#     my_test_forward
#     # clean up
#     rm -rf OUTPUT_FILES* src_rec_true.dat time.txt *h5 models
# elif [ "$TESTDIR" == "examples/4_forward_accuracy_3rd_cuthill_isotropic/" ]; then
#     my_test_forward
#     # clean up
#     rm -rf OUTPUT_FILES* src_rec_true.dat time.txt *h5 models
# elif [ "$TESTDIR" == "examples/5_use_different_types_of_data_in_model_update/" ]; then
#     my_test_forward_2
#     # clean up
#     rm -rf OUTPUT_FILES* src_rec_true.dat time.txt *h5 models
else
    echo "No test script for this example!"
    exit 1
fi

echo
echo "test finished successfully!"
echo `date`
echo