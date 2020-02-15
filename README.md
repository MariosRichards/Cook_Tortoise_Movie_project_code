# Cook_Tortoise_Movie_project_code


Notebook Index:
* Bunch of Regressions
    - hard fixes for dates
    - builds and saves "INT_df_ord_only"
    - INT_df_ord_only correlation pairs ranks
    - INT_df_ord_only Factor Analysis
    - INT_df_ord_only xgboost regression
    
* Create_cast_crew_aggregates_for_tmbd
    - load crew/cast and crew_/cast_individual data
    - create cast/crew aggregates <trait> x <aggregation method> (mean/lead/leadX/second)
    - get birth/death/lifespan data
    - hard fixes for birth/date data
    - birth/death rel to 1900/life_span added to aggregates
    - added aggregation type "leading_man" and "leading_woman"
    
* Dataset Inventory
    - lists (with comments) all input datasets and builds INT_df
    - contains hardfixes for the various incoming datasets
    
* EDA
    - hardcode fixes for SER_releaseDate and TMB_revenue
    - try to summarise "profitability" as gross-by-budget and normalise measure
    - use gender-guesser module to get gender_guess_integrated column for actors/crew
    - define "importulence" for cast/crew as credit_numer/mean_order
    - set up cast/crew aggregation, look at cast_size/crew_size dist
    - def "horizontal bar chart with N"
    - analysis
        - gross_by_budget for prod countries/companies
        - prod companies w *very high/very low* profitability all with only 1 movie
        - specific breakdown on top 15 examples
        - specific breakdown on bottom 15 examples
    - set up for IND_df_ord_only -> FA -> Quality/Exposure
    - xgboost regression
        - gross_by_budget_normalised
        - credit_number_cast_lead10
        - importulence_cast_lead10
        - gender_guess_integrated_cast_lead10
        - gender_guess_integrated_crew_lead5
        - mean_order_cast_lead10
        - fractional_mean_order_cast_lead10
        - gross_by_budget_normalised
        - release_year
        - TMB_runtime
        - gross
        - budget
        - PERS_agreeableness_r
        - PERS_openness_mean
        - TMB_vote_count
        - TMB_popularity_normalised
        - HETREC_rtAudienceScore
        
* Gender Balance and Repackaging TMBD API data
    - gender by order - lineplot/areaplot
    - gender balance over time (facetplot by decade)
    - gender by fractional order
    
    - code to get daily update datasets from tmb
    - notes and cleaning code for each individual tmb dataset
    - helper functions for dealing with large datasets
    - checking a lot of the tmb datastrctures (departments/jobs)
    - some cast/crew gender hard coding (where the two disagree on the same person)
    - hard fix on profile paths ("")
    - a *lot* of output TMB file processing!

* Genre Integration & Inferring Fusion Variables
    - SER_releaseDate hard fixes
    - load HETREC/ML/TMB genres
        - look at which categories are synonymous/need to be dropped
        - built int_genres
        - create scarcity-ordered single genres (movies have mult genres, pick the lowest freq on assumption that it's most specific)
    - try "target encoding" on profit by genre (appears, but below Year/Quality/Exposure)
    - genre x release_year are chart
    - genre x release_year facet chart
    - extract release day/month/decade/weekday and add to INT_df/INT_df_ord_only
    - extract and add collection_size
    - python wordcloud module tested on taglines by decade
    - xgboost release_year (very pred from credit_num of crew_lead5)
    - xgboost Exposure (NUM_domestic_gross_fraction, num_other_releases_that_year, release_year, Quality, gross_normalised, release_decade__2000 ... TMB_title_used_by_num_other_movies
    - xgboost Quality (Drama+,Horror--,runtime+,prod_USA-,PERS_0(?),Docu+,release_year(+/-),budget--,action-,
    - use WPCA to try turn all ratings columns into factors (looks like Quality/Exposure split)
    - xgboost tagline_length (num_other_releases_that_year---,release_year--:both mod by docu)
    

* Harmonise Dates and Genres and Runtimes
    - load INT_df, hardfix release dates (SER/NUM/TMB)
    - integrate release dates with reliability hierarchy
    - turn into datetimes, extract release_day/month/weekday/decade
    - handle dates with no month/day (e.g. 2001/1/1)
    - helper function to get years (float) from 1900 as a fraction
    - create a set of cast/crew aggregates for "age_at_release" and "years_until_death"
    - hard fix people showing up as having died before the movie began (who aren't, say, authors)
    - there's a problem with orders in cast - I'm guessing sometimes skips positions    - it's fixed and the original is saved in original_order, max_order also saved as TMB_crew/TMB_cast
    - traits from crew/cast_individuals saved into crew/cast_dfs
    - genre stuff - harmonised genre names for Science Fiction -> Sci-Fi, Family -> Children -> saved as INT_df_with_dates_and_genres
    ## (0) Release Year x Movie number (lineplot)
    - uId overlaps?
    - bi/tri module runtime distribution
    ## (1) Release Year x Runtime (lineplot/violinplot/ridgeplot)
    - create single_genres column 
    # Age at movie release for Leading Men/Women Cast/Crew by Decade
    # Years Until Death for Leading Men/Women Cast/Crew by decade
    # Total Life Span for Leading Men/Women Case/Crew by decade
    # Cast Lead Gender x Release Year
    # Leading Man Order x Release Year
    # Leading Woman Order x Release Year
    # Lead Man-Woman Age Gap x Release Year
    
* Inferring Factors
    - create INT_df_ord_only for factor analysis (also drop low sample size r_value from PERS)
    - Factor Analysis on ratings columns (nas mean filled) to get Quality and Exposure factors -> "ratings_fusion"
    
    - tried to boil PERS values down into a few factors (some inconclusive exploration of whether/where we should ditch entries based on the p values)
        - PERS_0 - high mean values for all big 5 (more extraversion)
        - PERS_1 - (conscientious/emo_stab)_mean++, (extraversion/openness)_mean-
        - PERS_2 - (consc/agree/emo_stab)_std+,(extra/open/agree)_r+
        - PERS_3 - high r values for all big 5 (more emo_stab)
        - all *very noisy*
        
    - tried to boil aggregate cast/crew variables (as they were then) down
        - comp 0: unspecialised crew, done a lot, cast also done quite a bit (job_number_crew_lead5)
        - comp 1: cast done a lot, crew specialised (importulence_cast_lead10)
        - comp 2: crew prominence (importulence_crew_lead5)
        - comp 3: gender balance crew, weaker gender balance cast (_lead5/10 heading/equal head)
        - comp 4: (mean_order_cast_mean) cast crew mean order/fractional mean order (opposite of importulence/department number)
        - comp 5: (fractional_mean_order_crew_lead5) prominent, male-skew crew and female skew cast (horror?)
        - comp 6: (fractional_mean_order_crew_lead5) prominent, female-skew crew and male skew cast (documentary???)
        - comp 7: (mean_order_cast_lead) low prominence female-skew cast who've been in a lot of movies
        - comp 8: (fractional_mean_order_cast_mean)    
        
* Integrate Datasets
    - first attempt to integrate originald datasets
    - starts with TMB_ data
    - gender_guesser module (98% agreement on "male"/"female" with imdb data, 78% for "mostly_male/female"
    - some original_language hard fixes
    - set up to run Bayesian Belief Network fitting
    - Hong Kong, Ireland, Australia ridiculously profitable prod_country (I assume because of Bruce Lee/ filming environment?)
    - Romania, Bulgaria super non-profitable
    - interactive widgets for looking at movies sorted by PERS ratings
    - amalgamating and reconciling movie links from different datasets
    - harmonising movie title styiles (Name (Year)) - helper functions add_yr/red_yr
    - matching NUM to movieId on title - > load of hard fixes!
    - add new columns to NUM - worldwide_gross_divided_by_budget, international_gross, domestic_gross_fraction
    - convert TMB_original/spoken language to harmonised type
    - merge into INT_df
    - Factor Analysis on INT_df
        - "Movie Size" TMB_crew_size
        - high budget/gross with small crews
        - Japanese language/production/jap prod companies vs English/USA
        - Hindi (and French/Swedish) vs English (and Japanese)
        - Hindi vs good (and French/Swedish)
        - SER variables
        - ratings vs French/Swedish
        - Hong Kong/cantonese Jackie Chan vs Swedish
    - XGBoost profitability (box-cox normalised)
        - belongs_to_collection, release_year, domestic_gross_faction, vote_average
    - xgboost_tuner/GridSearchCV to get better xgboost params
    - try to install gender_guesser/Genderize/gender - only gender_guesser works reliably
    - Factor Analysis on PERS variables
        - extraversion_std+,all big5_means+ (extra last)
        - extraversion_std+,consc_mean-,consc_std+,agree_/emo_stab_mean-
        - emo_stab_r++,consc_r+,agree_r+,agree_mean-
        - agree_mean++,consc_mean-,openness_std-,openness_mean+
        ... - extra_r+++,open_r++,agree_r+

* Movie_project_datasets
    - first pass at gathering movie datasets
    - pass over all datasets
    - a load of notes/hard fixes about *very* long movies/TV series mislabelled as movies
    - attempts to box-cox normalise distributions
    - PERS dataset - process and get_big5_corr helper function
    (interactive widget to look at movies liked by high big5 factor/whose liking correlates with the big5 factor

* Person_id_API_data_and_award_data
    - person_ids data from tmdb api
        - birth/deathday (incomplete), known_for_department, place_of_birth
    - 220k_awards_by_directors and 900_acclaimed_directors_awards
        - it's all by director not by movie
    
* Release_Year
    - decent commenting!
## (0) Release Year x Movie number
## (1) Release Year x Runtime (lineplot year/decade and 
## (2) Release Year x How Many (Other) Movies The (top 5) Crew Were/Will Go On To Be In
    - violinplot/boxplot/boxenplot/stripplot/ridgeplot
    - boxenplot cast/crew_lead credit_number - lineplot cast/crew_lead credit_number
    - mean_lifetime_credits_of_top_crew/cast
## (3) Release Year x Num Production Companies/Countries
## (4) Release Year x Gender Balance (lead cast/crew)
## (5) Release Year x Crew/Cast Size (lineplot, boxenplot - very skewed distribution! - lineplot of medians)
## (6) Release Year x Quality/Exposure
## (7) Release Year x Profitability/Domestic_gross_fraction
## (8) Release Year x TMB_tagline_length    
## (9) Release Year x release_day/release_month (heatmap/barchart)
## (10) Release Year x Genres!
    -xgboost release_year (credit_number_crew_lead5, runtime, prod_companies, mean_order_crew_lead5)

    
* the-numbers.com scraper
    - use BeautifulSoup python module to scrape the-numbers.com (https://www.the-numbers.com/movie/budgets/all)
    - scrap no, page no, release data, movie title, budget, domestic gross, worldwide gross


* TMDb API use
    - script to grab the latest version of the tmbd daily export files
    - the ids from the files are then used to make targeted calls to the APIs
    - using multiprocessing to efficiently make multiple simultaneous requests
    - hardfixes for death_dates

* TMDB_release_dates
    - complex tmdb release_date structure depends for each country
    - certification categories
    - type (premiere/theatrical (limited)/theatrical/digital/physical/tv
    - moderate attempts to harmonise all column values
    - create country__certificate pairs
    - looked at correlates (brazil/portugal ~.7, everything else <=.35)
    - factor analysis
        - AU__M (lots of age 12 cert)
        - GB__15 (lots of age 15 cert)
        - SE__7 (6/7 - GB parental guidance)
        - GB__18 (lots of age 18 cert)
        - GB__U (lots of universal cert)
        - Brazil/portugal
    - brief look at certification API (does not look straightforward)









