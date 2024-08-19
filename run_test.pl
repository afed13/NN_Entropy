#! /usr/bin/perl -w

$|=1;
$nproc = $ARGV[1] or die "Unknown number of threads\n";

$test1 = "./nn_entropy.exe $nproc  < c1.inp";
$test2 = "./nn_entropy.exe $nproc  < c2.inp";
$test3 = "./nn_entropy.exe $nproc  < c3.inp";
$test4 = "./nn_entropy.exe $nproc  < c4.inp";
$test5 = "./nn_entropy.exe $nproc  < c5.inp";


if($ARGV[0] == 1) {
    print "TESTING:\n$test1 \n\n";
    system(" $test1 ");
}
if($ARGV[0] == 2) {
    print "TESTING:\n$test2 \n\n";
    system(" $test2 ");
}
if($ARGV[0] == 3) {
    print "TESTING:\n$test3 \n\n";
    system(" $test3 ");
}
if($ARGV[0] == 4) {
    print "TESTING:\n$test4 \n\n";
    system(" $test4 ");
}
if($ARGV[0] == 5) {
    print "TESTING:\n$test5 \n\n";
    system(" $test5 ");
}

