Data Files
============
The data file should be a comma separated file with a single header line.  
Comment lines that begin with `#` will be ignored.  
The data columns should be arranged in the following order:  
  1. Sample ID: Any alpha numeric combination
  2. X or Latitude coordinate
  3. Y or Longitude coordinate
  4. Marker data

The alleles for each sample should be separated by `/` and each allele should be represented by an integer.  
Example for a diploid organism `123/123`  
Example for a tetraploid organism `12345/12345/12345/12345`  

Missing alleles should be indicated by any of the following:  
`None`, `Nan`, `Na`, `X`, `XX`, `XXX`, `0`, `00`, `000`, `-`,`.`  
Case does not matter.  
Because `0` represents a null allele, do not use `0` to code for actual alleles.  
