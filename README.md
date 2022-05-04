# PISA Revisited

## Summary 
The OCED’s Programme for International Student Assessment (PISA) has consistently found that girls outperform boys in reading, among other domains, and that this gender gap is large, worldwide, and persistent throughout primary and secondary schooling. Cited literature highlights how girls’ academic strength relative to their male peers may impact their confidence and interests across subjects, thus explaining differences in girls’ career aspirations, such as a lower likelihood of joining STEM fields.

This project uses a machine learning framework to identify the strongest predictors of reading scores from the complete 2018 PISA dataset for boys and girls. Using the SKLearn library for Python, a multiple regression was trained on the data and used baseline model, with subsequent models introducing a Ridge penalty, polynomial regression as well as a Random Forest Regressor. Regressions with a Ridge penalty, and polynomial performed worse than the baseline regression while an Extra Trees Regressor slightly improved on the results of the Random Forest algorithm.

### Methods

Data preprocessing and analysis was conducted using Python and the SciKit Learn library.

### Data
The underlying dataset of this project is based on the full student questionnaire of the [2018 iteration of the PISA](https://www.oecd.org/pisa/data/2018database/). The initial loading of the full dataset produced a pandas dataframe with 612,004 observations and 1,120 columns. In accordance with the results of a literature review and previous research, a selection of variables with high construct validity was conducted, including but not limited to, items related to self-efficacy, reading habits and attitudes, school environment, teacher interaction, and parental involvement. As a result, the final dataset included 205 variables as covariates and reading score as the independent variable. A sample of 100,000 observations was randomly created for further processing. 

### Contributors

- Anna Weronika Matysiak ([GitHub](https://https://github.com/AnnaWeronikaMatysiak))
- Johanna Mehler ([GitHub](https://https://github.com/j-mehler))
- Max Eckert ([GitHub](https://github.com/m-b-e), [twitter](https://twitter.com/mabrec1))


### Further Resources

- [Brow (2018): Significant predictors of mathematical literacy for top-tiered countries/economies, Canada, and the United States on PISA 2012: Case for the sparse regression model](https://doi.org/10.1111/bjep.12254)
- [Stoet & Gary (2018): The Gender-Equality Paradox in Science, Technology, Engineering, and Mathematics Education](https://doi.org/10.1177/0956797617741719)
- [Don & Hu (2018): An Exploration of Impact Factors Influencing Students’ Reading Literacy in Singapore with Machine Learning Approaches](https://doi.org/10.5539/ijel.v9n5p52)
- [Lezhnina & Kismihók (2022): Combining statistical and machine learning methods to explore German students’ attitudes towards ICT in PISA](https://doi.org/10.1080/1743727X.2021.1963226)
- [Lee (2022): What drives the performance of Chinese urban and rural secondary schools: A machine learning approach using PISA 2018](https://doi.org/10.1016/j.cities.2022.103609)

### License

The material in this repository is made available under the [MIT license](http://opensource.org/licenses/mit-license.php). 
