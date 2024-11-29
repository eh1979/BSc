# BSc
Repository for code written for my Bachelor's Project

Plotting code expects data in a specific folder structure:

```
sample_name/ -> VSM/ -> Data/ -> your_data.VHD
             -> XRD/ -> 2thetachi/ -> Data/ -> your_data.txt
                     -> 2theta/ -> Data/ -> your_data.txt
                     -> theta2theta/ -> Data/ -> your_data.ras
                     -> XRR/ -> Data/ -> your_data.txt
```

Can then use the ```run.sh``` file by changing ```BASE_FOLDER``` filepath to the file path that your data folders are in. Then after chmod to turn into .exe, run as 
```
./run.sh sample_name
```
Or for ```run_all.sh```, define ```BASE_FOLDER``` as before and change the ```SAMPLE_START``` variable from EH in line 6 to the beginning of chosen sample name, where sample_folders should be labelled in a similar format to EH01, EH01_350, EH02, etc... with a consistent sample start. Run with:
```
./run_all.sh
```
for standard graphs which show fitting methods, or:
```
./run_all.sh report folder_name 
```
for report standard graphs with comparisons between all data folders beginning with sample_start.  *Where the data folder created will then be called report_(given folder_name).*
