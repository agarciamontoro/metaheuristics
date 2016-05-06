#!/bin/bash

fileName=$1

sed -i '1i particiones, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12' $fileName
sed -i '2s/^/Partición 1-1,/' $fileName
sed -i '3s/^/Partición 1-2,/' $fileName
sed -i '4s/^/Partición 2-1,/' $fileName
sed -i '5s/^/Partición 2-2,/' $fileName
sed -i '6s/^/Partición 3-1,/' $fileName
sed -i '7s/^/Partición 3-2,/' $fileName
sed -i '8s/^/Partición 4-1,/' $fileName
sed -i '9s/^/Partición 4-2,/' $fileName
sed -i '10s/^/Partición 5-1,/' $fileName
sed -i '11s/^/Partición 5-2,/' $fileName
sed -i '12s/^/Medias,/' $fileName
