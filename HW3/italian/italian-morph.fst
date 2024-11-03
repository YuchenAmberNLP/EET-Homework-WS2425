%%% Single Character Symbols %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#letter# = [a-zA-ZàèéòìùÀÈÉÒÌÙ]

%%% Analysis Features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Adjectives ending with -e have only two forms, varying only according to number (singular/plural).
% number feature
#Number# = <sg><pl>

% gender feature
#Gender# = <masc><fem>

% Person Feature
#Person# = <1><2><3>

% degree feature
% #Degree# = <positive><comparative><superlative>

%%% Agreement Features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#WordArt# = <ADJ><VERB>

%%% #ADJClass# = <ADJ-o><ADJ-e>
%%% #VerbClass# = <Verb-are><Verb-ere><Verb-ire>

% verbal features
%%% There are 3 moods (Indicative Mood, Conditional mood, Subjunctive mood, Imperative mood) for verbs in Italian, for Indicative Mood, there are 4 tenses (present, imperfect, preterite, and future), for conditional mood, there is only 1 tense (present), for subjunctive mood, there are 2 tenses(present), and for imperative mood, there is only present tense conjunction.
#VerbTense# = <present><imperfect><preterite><future>
#VerbMood# = <indicative><conditional><subjunctive><imperative>
#VerbNominalForm# = <infinite><participle><gerund>

% Morphosyntactic Features
#MorphSyn# = #Number# #Gender# #Person# #VerbTense# #VerbMood# #VerbNominalForm#


#ALPJHABET# = #letter# \
              #Number#:<> #Gender#:<> #Person#:<> #Degree#:<> \
              #WordArt#:<> #ADJClass#:<> #VerbClass#:<> \
              #VerbTense#:<> #VerbMood#:<> #VerbNominalForm#:<>

%%% definition of the inflectional classes %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%-ADJEKTIVE-Regeln%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
$adjsstems-o$ = "adjectives.lex" || [a-z]* {o}:{} <>:<AdjReg-o>
$adjsstems-e$ = "adjectives.lex" || [a-z]* {e}:{} <>:<AdjReg-e>

$AdjReg-o$ = {<masc><sg>}:{o} |\
             {<masc><pl>}:{i} |\
             {<fem><sg>}:{a} |\
             {<fem><pl>}:{e}

$AdjReg-e$ = {<masc><sg>}:{e} |\
             {<masc><pl>}:{i} |\
             {<fem><sg>}:{e} |\
             {<fem><pl>}:{i}

%%% $INFL-ADJ$ = ($AdjGetStem-o$ || $AdjReg-o$) | ($AdjGetStem-e$ || $AdjReg-e$)
$INFL-ADJ$ = ($adjsstems-o$ || $AdjReg-o$) | ($adjsstems-e$ || $AdjReg-e$)

%%%%%%%%%-VERB-Regeln%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% $verbstems-are$ = "verbs.lex" || [a-z]* {are}:{} <>:<VERB-are>

$verbstems-are$ = #letter# + "are<VERB>" <=> __ <Verb-are>;
$VerbRegInd-are$ = {<indicative><present><1><sg>}:{o} |\
                {<indicative><present><2><sg>}:{i} |\
                {<indicative><present><3><sg>}:{a} |\
                {<indicative><present><1><pl>}:{imao} |\
                {<indicative><present><2><pl>}:{ate} |\
                {<indicative><present><3><pl>}:{ano} |\
                {<indicative><imperfect><1><sg>}:{avo} |\
                {<indicative><imperfect><2><sg>}:{avi} |\
                {<indicative><imperfect><3><sg>}:{ava} |\
                {<indicative><imperfect><1><pl>}:{avamo} |\
                {<indicative><imperfect><2><pl>}:{avate} |\
                {<indicative><imperfect><3><pl>}:{avano} |\
                {<indicative><preterite><1><sg>}:{ai} |\
                {<indicative><preterite><2><sg>}:{asti} |\
                {<indicative><preterite><3><sg>}:{ò} |\
                {<indicative><preterite><1><pl>}:{ammo} |\
                {<indicative><preterite><2><pl>}:{aste} |\
                {<indicative><preterite><3><pl>}:{arono} |\
                {<indicative><future><1><sg>}:{erò} |\
                {<indicative><future><2><sg>}:{erai} |\
                {<indicative><future><3><sg>}:{eremo} |\
                {<indicative><future><1><pl>}:{eremo} |\
                {<indicative><future><2><pl>}:{erete} |\
                {<indicative><future><3><pl>}:{eranno};

$VerbRegCond-are$ = {<conditional><present><1><sg>}:{erei} |\
                {<conditional><present><2><sg>}:{eresti} |\
                {<conditional><present><3><sg>}:{erebbe} |\
                {<conditional><present><1><pl>}:{eremmo} |\
                {<conditional><present><2><pl>}:{ereste} |\
                {<conditional><present><3><pl>}:{erebbero};

$VerbRegSubj-are$ = {<subjunctive><present><1><sg>}:{i} |\
                {<subjunctive><present><2><sg>}:{i} |\
                {<subjunctive><present><3><sg>}:{i} |\
                {<subjunctive><present><1><pl>}:{iamo} |\
                {<subjunctive><present><2><pl>}:{iate} |\
                {<subjunctive><present><3><pl>}:{ino} |\
                {<subjunctive><imperfect><1><sg>}:{assi} |\
                {<subjunctive><imperfect><2><sg>}:{assi} |\
                {<subjunctive><imperfect><3><sg>}:{asse} |\
                {<subjunctive><imperfect><1><pl>}:{assimo} |\
                {<subjunctive><imperfect><2><pl>}:{aste} |\
                {<subjunctive><imperfect><3><pl>}:{assero};

$VerbRegCondImp-are$ = {<imperative><present><2><sg>}:{a} |\
                {<imperative><present><3><sg>}:{i} |\
                {<imperative><present><1><pl>}:{iamo} |\
                {<imperative><present><2><pl>}:{ate} |\
                {<imperative><present><3><pl>}:{ino};


$VerbReg-are-Infl$ = $verbstems-are$ || ($VerbRegInd-are$ | $VerbRegCond-are$ | $VerbRegSubj-are$ | $VerbRegCondImp-are$)

$INFL$ = <AdjReg-o> $INFL-ADJ$ |\
         <AdjReg-e> $INFL-ADJ$ |\
         <VerbReg-are> $VerbReg-are-Infl$