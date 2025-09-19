import streamlit as st

import numpy as np
import pandas as pd
import read_cgm_data as rd

pages_master = {0:[":house: Home","app_launch.py"],
              1:[":information_source: Data Structure","pages/1_read_data.py"],
              2:[":file_cabinet: Import_Data","pages/2_import_data.py"],
              3:[":man: Explore Data","pages/3_explore_data.py"],
              4:[":couple: Cohort Data","pages/4_compare_data.py"],
              5:[":floppy_disk: Export Data","pages/5_export_data.py"],
}

if 'pages_master' not in st.session_state:
    st.session_state['pages_master'] = pages_master
if 'cgm_data' not in st.session_state:
    st.session_state['cgm_data']=None
if 'current_file' not in st.session_state:
    st.session_state['current_file']=None
if 'skip_rows' not in st.session_state:
    st.session_state['skip_rows'] = 1
if 'date_col' not in st.session_state:
    st.session_state['date_col'] = None
if 'date_col_idx' not in st.session_state:
    st.session_state['date_col_idx'] = None
if 'glucose_col' not in st.session_state:
    st.session_state['glucose_col'] = None
if 'glucose_col_idx' not in st.session_state:
    st.session_state['glucose_col_idx'] = None
if 'date_format' not in st.session_state:
    st.session_state['date_format'] = '%Y-%m-%d %H:%M:%S'
if 'header_row' not in st.session_state:
    st.session_state['header_row'] = 0
if 'pages_dict' not in st.session_state:
    st.session_state['pages_dict'] = {pages_master[0][0]:pages_master[0][1],
                                    pages_master[1][0]:pages_master[1][1],
                                    #"Test":"pages/test_page.py",
                                    }
if 'time_delta' not in st.session_state:
    st.session_state['time_delta'] = 5
if 'units' not in st.session_state:
    st.session_state['units']="mg"
if 'cohort_stats' not in st.session_state:
    st.session_state['cohort_stats'] = None


pages_dict = st.session_state['pages_dict']
rd.display_page_links(pages_dict)
st.sidebar.markdown("#### Video: [Getting Started](https://youtu.be/rywPXWpToQQ)")

st.markdown('# GVC-Calc')
body = ""
options = [":orange_book: About",":scroll: Documentation"]
select = st.sidebar.radio(label = "Select:",options=options)

if st.session_state['current_file'] is not None:
    st.sidebar.button("Restart Session",on_click=rd.initialize_session)
if select == options[0]:
    st.subheader("About")
    body = "GVC-Calc was developed to assist researchers with Continuous Glucose Monitoring (CGM) "
    body += "data by West Point's AI Data Engineering and Machine Learning (AIDE-ML) Center. "
    body += "The focus of the tool is to calculate as many glycemic metrics as possible while "
    body += "documenting the choices that were made during the computation of each. "
    body += "We focus on csv files and assisting the user with the ability to see and load their "
    body += "file / files into the system assuming all of the CGM files are structured the same."
    st.markdown(body)
    st.subheader("Files")
    body = "Each file used in this app should have data for a single participant spanning a single "
    body+= "or multiple mutually exclusive time periods consisting of more than two days. "
    body+= "The app can accept multiple csv files representing many "
    body+= "participants as long as the files have the same structure in terms of the following:\n "
    body+= "1) datetime format \n 2) datetime column number \n 3) glucose column number \n "
    body+= "4) time delta between measurements (1 min, 5 mins, or 15 mins) \n "
    body+= "5) unit of measurment of glucose level (mg/dL or mmol/L) \n "
    body+= "6) location of the header row \n 7) number of rows to skip before getting to the data. \n\n "
    body+= "Once the structure is understood, the user can import as many of those files as needed."
    st.markdown(body)
    img_link = "https://images.squarespace-cdn.com/content/v1/5be5c21e75f9ee21b5817cc2/"
    img_link +="cc5acf74-63ab-40f0-83b0-e67c484abf2b/example_csv_cgm_file.png?format=1500w"
    st.image(img_link,
             width=700,
             caption = "Figure 1: Example of CGM File.")
    st.subheader("Process")
    body = "The process for computing glycemic variability and control statistics for your files "
    body+= "is to help the computer understand your file structure, load them, analyze the files, "
    body+= "compare them with one another, and export the statistics for each file."
    
    st.markdown(body)
    img_link = "https://images.squarespace-cdn.com/content/v1/5be5c21e75f9ee21b5817cc2/"
    img_link+="12b0fc67-60fb-45ce-9228-9d64ae11bc17/process_2.png?format=2500w"
    st.image(img_link,
             width = 700,
             caption = "Figure 2: Process for Analyzing Multiple Files.")
    body = "After the structure is documented, you will be able to download all of the files that "
    body += "have the same structure and use the tool to analyze your data."
    st.markdown(body)
if select == options[1]:
    st.markdown("## Documentation")
    st.divider()
    tab_options = ["Data Display","Metrics","Interpretation","Challenges"]
    tabs = st.tabs(tab_options)
    with tabs[0]:
        
        st.markdown("##### ⚕️ How should the data be displayed")
        
        link = "- [Continuous glucose monitoring and metrics for clinical trials: an international consensus statement]"
        link += "(https://www.thelancet.com/journals/landia/article/PIIS2213-8587(22)00319-9/abstract)"
        link += "  \nBattelino(2023) \n\n"
        link+="""Randomised controlled trials and other prospective clinical studies for novel medical 
        interventions in people with diabetes have traditionally reported HbA1c as the measure of average 
        blood glucose levels for the 3 months preceding the HbA1c test date. The use of this measure 
        highlights the long-established correlation between HbA1c and relative risk of diabetes complications;
        the change in the measure, before and after the therapeutic intervention, is used by regulators for 
        the approval of medications for diabetes. However, with the increasing use of continuous glucose 
        monitoring (CGM) in clinical practice, prospective clinical studies are also increasingly using 
        CGM devices to collect data and evaluate glucose profiles among study participants, complementing
            HbA1c findings, and further assess the effects of therapeutic interventions on HbA1c. Data is 
            collected by CGM devices at 1–5 min intervals, which obtains data on glycaemic excursions and 
            periods of asymptomatic hypoglycaemia or hyperglycaemia (ie, details of glycaemic control that 
            are not provided by HbA1c concentrations alone that are measured continuously and can be analysed 
            in daily, weekly, or monthly timeframes). These CGM-derived metrics are the subject of 
            standardised, internationally agreed reporting formats and should, therefore, be considered 
            for use in all clinical studies in diabetes. The purpose of this consensus statement is to 
            recommend the ways CGM data might be used in prospective clinical studies, either as a specified 
            study endpoint or as supportive complementary glucose metrics, to provide clinical information 
            that can be considered by investigators, regulators, companies, clinicians, and individuals with 
            diabetes who are stakeholders in trial outcomes. In this consensus statement, we provide 
            recommendations on how to optimise CGM-derived glucose data collection in clinical studies, 
            including the specific glucose metrics and specific glucose metrics that should be evaluated. 
            These recommendations have been endorsed by the American Association of Clinical Endocrinologists, 
            the American Diabetes Association, the Association of Diabetes Care and Education Specialists, 
            DiabetesIndia, the European Association for the Study of Diabetes, the International Society 
            for Pediatric and Adolescent Diabetes, the Japanese Diabetes Society, and the Juvenile Diabetes 
            Research Foundation. :blue-background[A standardised approach to CGM data collection and reporting in clinical 
            trials will encourage the use of these metrics and enhance the interpretability of CGM data, 
            which could provide useful information other than HbA1c for informing therapeutic and treatment 
            decisions, particularly related to hypoglycaemia, postprandial hyperglycaemia, and 
            glucose variability. ]"""
        st.markdown(link)


        link = "- [Clinical Targets for Continuous Glucose Monitoring Data Interpretation:"
        link += " Recommendations From the International Consensus on Time in Range]"
        link += "(https://diabetesjournals.org/care/article/42/8/1593/36184/Clinical-Targets-for-Continuous-Glucose-Monitoring)"
        link += "  \nBattelino(2019) \n\n"
        link += """Improvements in sensor accuracy, greater convenience and ease of use, and expanding reimbursement 
        have led to growing adoption of continuous glucose monitoring (CGM). However, successful utilization of 
        CGM technology in routine clinical practice remains relatively low. This may be due in part to the lack 
        of clear and agreed-upon glycemic targets that both diabetes teams and people with diabetes can work toward.
        Although unified recommendations for use of key CGM metrics have been established in three separate 
        peer-reviewed articles, formal adoption by diabetes professional organizations and guidance in the 
        practical application of these metrics in clinical practice have been lacking. In February 2019, 
        the Advanced Technologies & Treatments for Diabetes (ATTD) Congress convened an international panel 
        of physicians, researchers, and individuals with diabetes who are expert in CGM technologies to 
        address this issue. :blue-background[This article summarizes the ATTD consensus recommendations for relevant aspects 
        of CGM data utilization and reporting among the various diabetes populations.]"""
        st.markdown(link)
        
        link = "- [Glucose Variability: A Review of Clinical Applications and Research Developments]"
        link += "(https://www.liebertpub.com/doi/full/10.1089/dia.2018.0092)"
        link += "-  \nRobard (2018) \n\n"
        link += """ Glycemic variability (GV) is a major consideration when evaluating quality of 
        glycemic control. GV increases progressively from prediabetes through advanced T2D and is 
        still higher in T1D. GV is correlated with risk of hypoglycemia. The most popular metrics 
        for GV are the %Coefficient of Variation (%CV) and standard deviation (SD). The %CV is 
        correlated with risk of hypoglycemia. Graphical display of glucose by date, time of day, 
        and day of the week, and display of simplified glucose distributions showing % of time in 
        several ranges, provide clinically useful indicators of GV. SD is highly correlated with 
        most other measures of GV, including interquartile range, mean amplitude of glycemic excursion, 
        mean of daily differences, and average daily risk range. Some metrics are sensitive to the 
        frequency, periodicity, and complexity of glycemic fluctuations, including Fourier analysis, 
        periodograms, frequency spectrum, multiscale entropy (MSE), and Glucose Variability 
        Percentage (GVP). Fourier analysis indicates progressive changes from normal subjects to 
        children and adults with T1D, and from prediabetes to T2D. The GVP identifies novel 
        characteristics for children, adolescents, and adults with type 1 diabetes and for adults 
        with type 2. GVP also demonstrated small rapid glycemic fluctuations in people with T1D 
        when using a dual-hormone closed-loop control. MSE demonstrated systematic changes from 
        normal subjects to people with T2D at various stages of duration, intensity of therapy, 
        and quality of glycemic control. We describe new metrics to characterize postprandial 
        excursions, day-to-day stability of glucose patterns, and systematic changes of patterns 
        by day of the week. Metrics for GV should be interpreted in terms of percentiles and 
        z-scores relative to identified reference populations. :blue-background[There is a need for large accessible 
        databases for reference populations to provide a basis for automated interpretation of 
        GV and other features of continuous glucose monitoring records.]"""
        st.markdown(link)

        link = "- :star: [Metrics for glycaemic control — from HbA1c to continuous glucose monitoring]"
        link += "(https://www.nature.com/articles/nrendo.2017.3)"
        link += "  \nKovatchev(2017) \n\n"
        link += """As intensive treatment to lower levels of HbA1c characteristically results in an increased 
        risk of hypoglycaemia, patients with diabetes mellitus face a life-long optimization problem to reduce 
        average levels of glycaemia and postprandial hyperglycaemia while simultaneously avoiding hypoglycaemia. 
        This optimization can only be achieved in the context of lowering glucose variability. In this Review, 
        I discuss topics that are related to the assessment, quantification and optimal control of glucose 
        fluctuations in diabetes mellitus. I focus on markers of average glycaemia and the utility and/or 
        shortcomings of HbA1c as a 'gold-standard' metric of glycaemic control; the notion that glucose 
        variability is characterized by two principal dimensions, amplitude and time; measures of glucose 
        variability that are based on either self-monitoring of blood glucose data or continuous glucose 
        monitoring (CGM); and the control of average glycaemia and glucose variability through the use of 
        pharmacological agents or closed-loop control systems commonly referred to as the 'artificial pancreas'. 
        :blue-background[I conclude that HbA1c and the various available metrics of glucose variability reflect the management 
        of diabetes mellitus on different timescales, ranging from months (for HbA1c) to minutes (for CGM). 
        Comprehensive assessment of the dynamics of glycaemic fluctuations is therefore crucial for 
        providing accurate and complete information to the patient, physician, automated decision-support 
        or artificial-pancreas system.]"""
        st.markdown(link)
        
        link = "- [International Consensus on Use of Continuous Glucose Monitoring]"
        link += "(https://diabetesjournals.org/care/article/40/12/1631/37000/International-Consensus-on-Use-of-Continuous)"
        link += "  \nDanne(2017) \n\n"
        link += """Measurement of glycated hemoglobin (HbA1c) has been the traditional method for assessing 
        glycemic control. However, it does not reflect intra- and interday glycemic excursions that may lead 
        to acute events (such as hypoglycemia) or postprandial hyperglycemia, which have been linked to both 
        microvascular and macrovascular complications. Continuous glucose monitoring (CGM), either from real-time 
        use (rtCGM) or intermittently viewed (iCGM), addresses many of the limitations inherent in HbA1c testing 
        and self-monitoring of blood glucose. Although both provide the means to move beyond the HbA1c measurement 
        as the sole marker of glycemic control, standardized metrics for analyzing CGM data are lacking. Moreover, 
        clear criteria for matching people with diabetes to the most appropriate glucose monitoring methodologies, 
        as well as standardized advice about how best to use the new information they provide, have yet to be 
        established. In February 2017, the Advanced Technologies & Treatments for Diabetes (ATTD) Congress 
        convened an international panel of physicians, researchers, and individuals with diabetes who are 
        expert in CGM technologies to address these issues. :blue-background[This article summarizes the ATTD consensus 
        recommendations and represents the current understanding of how CGM results can affect outcomes.]"""
        st.markdown(link)

        link = "- [Statistical Tools to Analyze Continuous Glucose Monitor Data]"
        link +="(https://www.liebertpub.com/doi/abs/10.1089/dia.2008.0138) "
        link += "  \nClarke(2009) \n\n"
        link += """Continuous glucose monitors (CGMs) generate data streams that are both complex and 
        voluminous. The analyses of these data require an understanding of the physical, biochemical, 
        and mathematical properties involved in this technology. This article describes several methods 
        that are pertinent to the analysis of CGM data, taking into account the specifics of the 
        continuous monitoring data streams. These methods include: (1) evaluating the numerical and 
        clinical accuracy of CGM. We distinguish two types of accuracy metrics—numerical and clinical—each 
        having two subtypes measuring point and trend accuracy. The addition of trend accuracy, e.g., 
        the ability of CGM to reflect the rate and direction of blood glucose (BG) change, is unique to 
        CGM as these new devices are capable of capturing BG not only episodically, but also as a process 
        in time. (2) Statistical approaches for interpreting CGM data. The importance of recognizing 
        that the basic unit for most analyses is the glucose trace of an individual, i.e., a 
        time-stamped series of glycemic data for each person, is stressed. We discuss the use of 
        risk assessment, as well as graphical representation of the data of a person via glucose 
        and risk traces and Poincaré plots, and at a group level via Control Variability-Grid Analysis. 
        :blue-background[In summary, a review of methods specific to the analysis of CGM data series is presented, 
        together with some new techniques. These methods should facilitate the extraction of information 
        from, and the interpretation of, complex and voluminous CGM time series.]"""
        st.markdown(link)

        link="- [Statistical Packages and Algorithms for the Analysis of Continuous Glucose Monitoring Data: A Systematic Review]"
        link+="(https://journals.sagepub.com/doi/full/10.1177/19322968231221803) "
        link+="  \nOlsen(2024) \n\n"
        link+="""Results: A total of 8731 references were screened and 46 references were included. 
        We identified 23 statistical packages for the analysis of CGM data. The statistical packages 
        could calculate many metrics of the 2022 CGM consensus and non-consensus CGM metrics, and 
        22/23 (96%) statistical packages were freely available. Also, 23 statistical algorithms were 
        identified. The statistical algorithms could be divided into three groups based on content: 
        (1) CGM data reduction (eg, clustering of CGM data), (2) composite CGM outcomes, and (3) other 
        CGM metrics."""
        st.markdown(link)

    with tabs[1]:
        st.markdown("##### ⚕️ How are the metrics calculated")

        link = "- :star: [Glycemic Variability Measures]"
        link += "(https://shiny.biostat.umn.edu/GV/README2.pdf)"
        link += "  \nEasyGV - Olawsky(2019)"
        st.markdown(link)
        
        link = "- [The M-Value, an Index of Blood-sugar Control in Diabetics]"
        link += "(https://onlinelibrary.wiley.com/doi/abs/10.1111/j.0954-6820.1965.tb01810.x)"
        link += "  \nM-value - Schlichtkrull(1965)"
        st.markdown(link)

        link = "- [Mean Amplitude of Glycemic Excursions, a Measure of Diabetic Instability]"
        link += "(https://diabetesjournals.org/diabetes/article/19/9/644/3599/Mean-Amplitude-of-Glycemic-Excursions-a-Measure-of)"
        link += "  \nMAGE - Service(1970)"
        st.markdown(link)

        link = "- [Day-to-day variation of continuously monitored glycaemia: A further measure of diabetic instability]"
        link += "(https://link.springer.com/article/10.1007/BF01218495)"
        link += "  \nMODD - Molnar(1972)"
        st.markdown(link)

        link = "- [“J”-Index. A New Proposition of the Assessment of Current Glucose Control in Diabetic Patients]"
        link += "(https://www.thieme-connect.com/products/ejournals/abstract/10.1055/s-2007-979906)"
        link += "  \nJ-Index - Wojcicki(1995)"
        st.markdown(link)

        link = "- [Assessment of the Severity of Hypoglycemia and Glycemic Lability in Type 1 Diabetic Subjects Undergoing Islet Transplantation]"
        link +="(https://diabetesjournals.org/diabetes/article/53/4/955/24293/Assessment-of-the-Severity-of-Hypoglycemia-and)"
        link +="  \nLability Index - Ryan(2004)"
        st.markdown(link)

        link = "- [A Novel Approach to Continuous Glucose Analysis Utilizing Glycemic Variation]"
        link += "(https://www.liebertpub.com/doi/abs/10.1089/dia.2005.7.253)"
        link += "  \nCONGA - McDonnell(2005)"
        st.markdown(link)

        link = "- [Evaluation of a New Measure of Blood Glucose Variability in Diabetes ]"
        link += "(https://diabetesjournals.org/care/article/29/11/2433/24571/Evaluation-of-a-New-Measure-of-Blood-Glucose)"
        link += "  \nADRR - Kovatchev(2006)"
        st.markdown(link)

        link = "- [A Novel Analytical Method for Assessing Glucose Variability: Using CGMS in Type 1 Diabetes Mellitus]"
        link += "(https://www.liebertpub.com/doi/10.1089/dia.2006.8.644)"
        link += "  \nLGBI/HGBI - McCall(2006)"
        st.markdown(link)

        link = "- [A method for assessing quality of control from glucose profiles]"
        link += "(https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1464-5491.2007.02119.x)"
        link += "  \nGRADE - Hill(2007)"
        st.markdown(link)

        link = "- [Translating the A1C Assay Into Estimated Average Glucose Values]"
        link+="(https://diabetesjournals.org/care/article/31/8/1473/28589/Translating-the-A1C-Assay-Into-Estimated-Average)"
        link+="  \neA1C - Nathan(2008)"
        st.markdown(link)

        link = "- [Measure for Measure; Consequences, Detection and Treatment of Hyperglycaemia]"
        link += "(https://www.researchgate.net/profile/Jeroen-Hermanides/publication/"
        link += "5520413_No_Apparent_Local_Effect_of_Insulin_on_Microdialysis_Continuous"
        link += "_Glucose-_Monitoring_Measurements/links/5406f7ec0cf2c48563b28626/"
        link += "No-Apparent-Local-Effect-of-Insulin-on-Microdialysis-Continuous-Glucose-"
        link += "Monitoring-Measurements.pdf)"
        link += "  \nMAG - Hermanides(2010)"
        st.markdown(link)

        link = "- [Glucose Management Indicator (GMI):A New Term for Estimating A1C From Continuous "
        link += "Glucose Monitoring]"
        link += "(https://doi.org/10.2337/dc18-1581)"
        link += "-  \nGMI - Begenstal(2018)"
        st.markdown(link)

        link = "- [Glycemic Variability Percentage: A Novel Method for Assessing Glycemic Variability"
        link += "from Continuous Glucose Monitor Data]"
        link += "(https://www.liebertpub.com/doi/full/10.1089/dia.2017.0187)"
        link += "-  \nGVP - Peyser(2018)"
        st.markdown(link)

        link = "- [Evaluating Glucose Control With a Novel Composite Continuous Glucose "
        link += "Monitoring Index]"
        link += "(https://journals.sagepub.com/doi/full/10.1177/1932296819838525)"
        link += "  \nCOGI - Leelarathna(2020)"
        st.markdown(link)

        link = "- [A Glycemia Risk Index (GRI) of Hypoglycemia and Hyperglycemia for "
        link +="Continuous Glucose Monitoring Validated by Clinician Ratings]"
        link += "(https://journals.sagepub.com/doi/full/10.1177/19322968221085273)"
        link += "  \nGRI - Klonoff(2023)"
        st.markdown(link)

        link = "- [Do Metrics of Temporal Glycemic Variability Reveal Abnormal Glucose "
        link += "Rates of Change in Type 1 Diabetes?]"
        link += "(https://journals.sagepub.com/doi/10.1177/19322968241298248)"
        link += "  \nTIF - Richardson(2024)"
        st.markdown(link)

    with tabs[2]:
        st.markdown("##### ⚕️ What is normal?")

        link = "- [Normal Reference Range for Mean Tissue Glucoseand Glycemic Variability Derived from Continuous "
        link += "Glucose Monitoring for Subjects Without Diabetes in Different Ethnic Groups]"
        link += "(https://pmc.ncbi.nlm.nih.gov/articles/PMC3160264/pdf/dia.2010.0247.pdf)"
        link += "  \nHill (2011) \n\n"
        link += """Results: Eight CGM traces were excluded because there were inadequate data. 
        From the remaining 70 traces, normative reference ranges (mean +- 2 SD) for glycemic 
        variability were calculated: SD, 0-3.0; CONGA, 3.6-5.5; LI, 0.0-4.7; J-Index, 4.7-23.6; 
        LBGI, 0.0-6.9; HBGI, 0.0-7.7; GRADE, 0.0-4.7; MODD, 0.0-3.5; MAGE-CGM, 0.0-2.8; 
        ADDR, 0.0-8.7; M-value, 0.0-12.5; and MAG, 0.5-2.2."""
        st.markdown(link)

        link = "- [CGMap: Characterizing continuous glucose monitor data in thousands of non-diabetic individuals]"
        link += "(https://www.cell.com/cell-metabolism/fulltext/S1550-4131(23)00129-8)"
        link += "  Keshet (2023) \n\n"
        link += """Despite its rising prevalence, diabetes diagnosis still relies on measures 
        from blood tests. Technological advances in continuous glucose monitoring (CGM) devices 
        introduce a potential tool to expand our understanding of glucose control and variability 
        in people with and without diabetes. Yet CGM data have not been characterized in large-scale 
        healthy cohorts, creating a lack of reference for CGM data research. Here we present 
        CGMap, a characterization of CGM data collected from over 7,000 non-diabetic 
        individuals, aged 40–70 years, between 2019 and 2022. We provide reference values 
        of key CGM-derived clinical measures that can serve as a tool for future CGM research. 
        We further explored the relationship between CGM-derived measures and diabetes-related 
        clinical parameters, uncovering several significant relationships, including 
        associations of mean blood glucose with measures from fundus imaging and sleep 
        monitoring. These findings offer novel research directions for understanding the 
        influence of glucose levels on various aspects of human health."""

        st.markdown(link)
    with tabs[3]:
        st.markdown("##### ⚕️ Challenges")

        link = "- [The Challenges of Measuring Glycemic Variability]"
        link += "(https://journals.sagepub.com/doi/abs/10.1177/193229681200600328)"
        link += "  \nRodbard (2012) \n\n"
        link += """This commentary reviews several of the challenges encountered when 
        attempting to quantify glycemic variability and correlate it with risk of diabetes 
        complications. These challenges include (1) immaturity of the field, including 
        problems of data accuracy, precision, reliability, cost, and availability; 
        (2) larger relative error in the estimates of glycemic variability than in the 
        estimates of the mean glucose; (3) high correlation between glycemic variability 
        and mean glucose level; (4) multiplicity of measures; (5) correlation of the 
        multiple measures; (6) duplication or reinvention of methods; (7) confusion of 
        measures of glycemic variability with measures of quality of glycemic control; 
        (8) the problem of multiple comparisons when assessing relationships among multiple 
        measures of variability and multiple clinical end points; and (9) differing needs 
        for routine clinical practice and clinical research applications.
                """
        st.markdown(link)

        link = "- [Glucose Variability - Service]"
        link += "(https://diabetesjournals.org/diabetes/article/62/5/1398/42890/Glucose-Variability)"
        link += "  \nService (2013) \n\n"
        link += """ The proposed contribution of glucose variability to the development 
        of the complications of diabetes beyond that of glycemic exposure is supported 
        by reports that oxidative stress, the putative mediator of such complications, 
        is greater for intermittent as opposed to sustained hyperglycemia. Variability 
        of glycemia in ambulatory conditions defined as the deviation from steady state 
        is a phenomenon of normal physiology. Comprehensive recording of glycemia is 
        required for the generation of any measurement of glucose variability. To avoid 
        distortion of variability to that of glycemic exposure, its calculation should be 
        devoid of a time component."""
        st.markdown(link)

        link = "- [Glucose Variability: Timing, Risk Analysis, and Relationship to Hypoglycemia in Diabetes]"
        link += "(https://diabetesjournals.org/care/article/39/4/502/28914/Glucose-Variability-Timing-Risk-Analysis-and)"
        link += "  \nKovatchev (2016) \n\n"
        link += """ Glucose control, glucose variability (GV), and risk for hypoglycemia are 
        intimately related, and it is now evident that GV is important in both the physiology 
        and pathophysiology of diabetes. However, its quantitative assessment is complex because 
        blood glucose (BG) fluctuations are characterized by both amplitude and timing. Additional 
        numerical complications arise from the asymmetry of the BG scale. In this Perspective, 
        we focus on the acute manifestations of GV, particularly on hypoglycemia, and review 
        measures assessing the amplitude of GV from routine self-monitored BG data, as well as 
        its timing from continuous glucose monitoring (CGM) data. With availability of CGM, the 
        latter is not only possible but also a requirement—we can now assess rapid glucose 
        fluctuations in real time and relate their speed and magnitude to clinically relevant outcomes. 
        Our primary message is that diabetes control is all about optimization and balance between 
        two key markers—frequency of hypoglycemia and HbA1c reflecting average BG and primarily 
        driven by the extent of hyperglycemia. GV is a primary barrier to this optimization, 
        including to automated technologies such as the “artificial pancreas.” Thus, it is time to 
        standardize GV measurement and thereby streamline the assessment of its two most 
        important components—amplitude and timing."""
        st.markdown(link)



