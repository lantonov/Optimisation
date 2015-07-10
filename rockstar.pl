#!C:\perl\bin\perl

# ROCK* Tuner
# Copyright (C) 2009-2015 Lyudmil Antonov and Joona Kiiski
#
# ROCK* Tuner is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ROCK* Tuner is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#use diagnostics;
use strict;
use warnings;

use threads;
use threads::shared;
use Time::HiRes qw(time);
use IPC::Open2;
use IO::Select;
use Config::Tiny;
use Text::CSV;

#use Math::Complex;
use Math::CDF qw(qbeta);
#use Math::Random qw(random_multivariate_normal);
use Math::Round qw(nearest nearest_floor);
use Math::MatrixReal;
use Statistics::Descriptive;
use Statistics::Distributions qw(chisqrdistr);
use List::Util qw(min max sum);
use IO::Handle;
use AutoLoader qw(AUTOLOAD);

STDOUT->autoflush(1);
STDERR->autoflush(1);

my $IS_WINDOWS = ($^O eq 'MSWin32');

### SECTION. Settings (Static data during execution)

my $ConfigFile = $ARGV[0] || die "You must pass the name of config file as parameter!";
my $Config = Config::Tiny->new;
$Config = Config::Tiny->read($ConfigFile) || die "Unable to read configuration file '" . $ConfigFile . "'";

my $simulate           = $Config->{Main}->{Simulate}           ; defined($simulate)          || die "Simulate not defined!";
my $variables_path     = $Config->{Main}->{Variables}          ; defined($variables_path)    || die "Variables not defined!";
my $log_path           = $Config->{Main}->{Log}                ; defined($log_path)          || die "Log not defined!";
my $gamelog_path       = $Config->{Main}->{GameLog}            ; defined($gamelog_path)      || die "GameLog not defined!";
my $iterations         = $Config->{Main}->{Iterations}         ; defined($iterations)        || die "Iterations not defined!";
				            
### SECTION. Strategic parameters for ROCK*

my $sigma              = $Config->{Main}->{Sigma}              ; defined($sigma)             || die "Sigma not defined!";
my $n_parameters       = $Config->{Main}->{n_parameters}       ; defined($n_parameters)      || die "n_parameters not defined!";
my $lambda             = $Config->{Main}->{lambda}             ; defined($lambda)            || die "lambda not defined!";
my $lambdaMD           = $Config->{Main}->{lambdaMD}           ; defined($lambdaMD)          || die "lambdaMD not defined!";
my $games_in_match     = $Config->{Main}->{games_in_match}     ; defined($games_in_match)    || die "games_in_match not defined!";
my $sigma_cur          = $Config->{Main}->{sigma_cur}          ; defined($sigma_cur)         || die "sigma_cur not defined!";
my $near_bins_size     = $Config->{Main}->{near_bins_size}     ; defined($near_bins_size)    || die "near_bins_size not defined!";
my $determinant        = $Config->{Main}->{determinant}        ; defined($determinant)       || die "determinant not defined!";
my $best_cost          = $Config->{Main}->{best_cost}          ; defined($best_cost)         || die "best_cost not defined!";
my $matches_evaluated  = $Config->{Main}->{matches_evaluated}  ; defined($matches_evaluated) || die "matches_evaluated not defined!";
my $initial_exp        = $Config->{Main}->{initial_exp}        ; defined($initial_exp)       || die "initial_exp not defined!";
my $initial_sd         = $Config->{Main}->{initial_sd}         ; defined($initial_sd)        || die "initial_sd not defined!";
my $constraint_penalty = $Config->{Main}->{constraint_penalty} ; defined($constraint_penalty)|| die "constraint_penalty not defined!";

### SECTION. Game parameters for ROCK* (same as in SPSA)

my $eng1_path          = $Config->{Engine}->{Engine1}          ; defined($eng1_path)         || $simulate || die "Engine1 not defined!";
my $eng2_path          = $Config->{Engine}->{Engine2}          ; defined($eng2_path)         || $simulate || die "Engine2 not defined!";
my $epd_path           = $Config->{Engine}->{EPDBook}          ; defined($epd_path)          || $simulate || die "EPDBook not defined!";
my $base_time          = $Config->{Engine}->{BaseTime}         ; defined($base_time)         || $simulate || die "BaseTime not defined!";
my $inc_time           = $Config->{Engine}->{IncTime}          ; defined($inc_time)          || $simulate || die "IncTime not defined!";
my $threads            = $Config->{Engine}->{Concurrency}      ; defined($threads)           || $simulate || die "Concurrency not defined!";
my $draw_score_limit   = $Config->{Engine}->{DrawScoreLimit}   ; defined($draw_score_limit)  || $simulate || die "DrawScoreLimit not defined!";
my $draw_move_limit    = $Config->{Engine}->{DrawMoveLimit}    ; defined($draw_move_limit)   || $simulate || die "DrawMoveLimit not defined!";
my $win_score_limit    = $Config->{Engine}->{WinScoreLimit}    ; defined($win_score_limit)   || $simulate || die "WinScoreLimit not defined!";
my $win_move_limit     = $Config->{Engine}->{WinMoveLimit}     ; defined($win_move_limit)    || $simulate || die "WinMoveLimit not defined!";
					  										    						        
$threads = 1 if ($simulate);								    						        

### SECTION. Variable CSV-file columns. (Static data during execution)
my $VAR_NAME      = 0; # Name
my $THETA         = 1; # Start Value (theta_0)
my $POLICY        = 2; # Policy
my $COST          = 3; # Cost
my $VAR_SIMUL_ELO = 4; # Simulation: Elo loss from 0 (optimum) to +-100)
my $VAR_END       = 5; # Nothing

# Extra calculated COLUMNS (SPSA parameters)
my $VAR_C         = 6; # c
my $VAR_A_END     = 7; # a in the last iteration
my $VAR_A         = 8; # a

### SECTION. Variable definitions. (Static data during execution)
my @variables;
my %variableIdx;

### SECTION. Log file handle (Static data during execution)
local (*LOG);

### SECTION. Shared data (volatile data during the execution)
my $shared_lock              :shared;
my $shared_iter              :shared; # Iteration counter
my $shared_policy_old        :shared; # Current policy by variable name
my $shared_mean_minus_min    :shared; # Current policy by variable name
my %shared_theta             :shared; # Current policy by variable name
my $var_eng1                 :shared;


### SECTION. Helper functions

# Function to safely read in a standard CSV-file.
sub read_csv
{
    my ($csvfile, $rows) = @_;
    my ($CSV, $row);

    open(INFILE, '<', $csvfile) || die "Could not open file '$csvfile' for reading!";
    binmode(INFILE);

    $CSV = Text::CSV->new();
    while($row = $CSV->getline(\*INFILE))
    {
     push(@$rows, $row);
    }

    $CSV->eof || die "CSV-file parsing error: " . $CSV->error_diag();
    close(INFILE);
}

# STEP. Calculated strategic ROCK* parameters.
my $cost2policy_cov_factor = chisqrdistr($n_parameters,0.05) * -0.5 / log($lambda);
my $range                  = chisqrdistr($n_parameters,0.05) * $lambdaMD;
my $expansion_factor_sigma = 1.3 ** (1 / log($n_parameters + 2.5)) - 1;
my $cc                     = 3 / ($n_parameters + 6) / log($n_parameters + 6);
my $ccov                   = 6 / ($n_parameters + 7) / log($n_parameters + 7);
my $chiN                   = $n_parameters ** 0.5 * (1 - 1 / (4 * $n_parameters) + 1 / (21 * $n_parameters ** 2));
my $imp_factor             = 1.3;
my $best_policy;

# STEP. Initial ROCK* zero vectors and matrices.
my $initial_theta   = new Math::MatrixReal(1,$n_parameters);
my $initial_policy  = new Math::MatrixReal(1,$n_parameters);
my $cost_history    = new Math::MatrixReal(1,$iterations);
my $theta_history   = new Math::MatrixReal($iterations,$n_parameters);
my $policy_history  = new Math::MatrixReal($iterations,$n_parameters);

# STEP. Init random generator
srand((time() ^ (time() % $])) ^ exp(length($0))**$$);

### SECTION. Log file handle (Static data during execution)
local (*LOG);

### SECTION. Execution preparation code ("main" function)
{
	my $row;

    # STEP. Open shared log file.
    open(LOG, '>', $log_path) || die "Could not open file '$log_path' for writing!";
    LOG->autoflush(1);

    # STEP. Read in variable data.
    read_csv($variables_path, \@variables);

    # STEP. Validate variable data.
    foreach $row (@variables)
    {
        die "Wrong number of columns!" if (scalar(@$row) != $VAR_END);

	    die "Invalid name: '$row->[$VAR_NAME]'"               if ($row->[$VAR_NAME]      !~ /^\w+$/);
        die "Invalid current: '$row->[$THETA]'"               if ($row->[$THETA]         !~ /^[-+]?[0-9]*\.?[0-9]+$/);
        die "Invalid max: '$row->[$COST]'"                    if ($row->[$COST]          !~ /^[-+]?[0-9]*\.?[0-9]+$/);
        die "Invalid min: '$row->[$POLICY]'"                  if ($row->[$POLICY]        !~ /^[-+]?[0-9]*\.?[0-9]+$/);
        die "Invalid simul ELO: '$row->[$VAR_SIMUL_ELO]'"     if ($row->[$VAR_SIMUL_ELO] !~ /^[-+]?[0-9]*\.?[0-9]+$/);
	}


    # STEP. Prepare shared data
    $shared_iter = 0;
    
    # STEP. Launch ROCK* threads
    my @thr;

    for (my $i = 1; $i <= $threads; $i++)
    {
        $thr[$i] = threads->create(\&run_rockstar, $i);

        # HACK: Under Windows the combination of starting new threads and 
        # calling open2() at the same time seems to be problematic.
        # So wait for 3 seconds to make sure each new thread has cleanly 
        # started the engine process before starting a new thread.
        sleep(3) if $IS_WINDOWS;
    }

    # STEP. Join threads
    for (my $i = 1; $i <= $threads; $i++)
    {
        $thr[$i]->join();
    }

    # STEP. Close Log file
    close(LOG);

    # STEP. Quit
    exit 0;
}

### SECTION. ROCK*
local (*GAMELOG);

sub run_rockstar
{
    my ($threadId) = @_;
    my $row;
	my $covar                  = new Math::MatrixReal($n_parameters,$n_parameters);
	$covar                     = $covar->one();
    my $covar_inv              = $covar->inverse();
    my $pc                     = new Math::MatrixReal($n_parameters,1);
    my $c_normalized           = $covar;

    # STEP. Open thread specific log file
    my $path = $gamelog_path;
    my $from = quotemeta('$THREAD');
    my $to = $threadId;
    $path =~ s/$from/$to/g;

    open(GAMELOG, '>', $path) || die "Could not open file '$path' for writing!";

    # STEP. Init engines
    engine_init() if (!$simulate);

# Generate random vector and initialise engine variables
#$my $random   = Math::MatrixReal->new_random(1,$n_parameters,{bounded_by=>[-1,1]});
my $policy   = $initial_policy;# + $random; # print "$policy\n";
#$theta       = $iter > 1 ? $theta : $initial_theta;
#my $var_eng1 = $initial_theta + $policy * $sigma;
my ($cost, %var_eng2, %var_value, $shared_theta2, $var_eng1);
my $theta = $initial_theta;

	# STEP. Create variable index for easy access.
    my $count = 1;
    foreach $row (@variables)
    {   
        $variableIdx{$row->[$VAR_NAME]} = $row;
        $shared_theta{$row->[$VAR_NAME]} = $row->[$THETA];
		$initial_theta->assign(1,$count,$row->[$THETA]);
		$count++;
    }

      # STEP. Calculate the names and variables for the second engine.
             foreach $row (@variables)
             {
                 my $name  = $row->[$VAR_NAME];
                 $var_value{$name}  = $shared_theta{$name};
                 $var_eng2{$name} = $var_value{$name};
             }

    while(1)
    {
        # SPSA coefficients indexed by variable.
#        my ($theta, $cost, $var_eng1, %var_eng2);

    my $iter;
	my $optimization_done = 0;

		{
             lock($shared_lock);

            # STEP. Increase the shared interation shared_iter
             if (++$shared_iter > $iterations)
             {
                 engine_quit() if (!$simulate);
                 return;
             }

         $iter = $shared_iter; # print "$iter \n";

 my $random   = Math::MatrixReal->new_random(1,$n_parameters,{bounded_by=>[-1,1]});
 my $policy_eps   = $policy + $random; # print "$policy_eps\n";   
	$var_eng1 = $theta + $policy_eps * $sigma;

    # STEP. Play two games (with alternating colors) and obtain the score from eng1 perspective.
        $cost = ($simulate ? simulate_2games(\$var_eng1, \%var_eng2) : engine_2games(\%var_eng2));
        $theta = $var_eng1;# print "$theta \n"; 
#        $cost = -$cost;

    # Record the policy, theta, and cost
	$policy_history->assign_row($iter,$policy);#  print "$policy_history \n";
	$theta_history->assign_row($iter,$theta); # print "$theta_history \n"; 
	$cost_history->assign(1,$iter,$cost);#	print "$cost_history \n";

	if ($cost < $best_cost) {
	      $best_cost   = $cost;
	      $best_policy = $policy_history->row($iter);# print "$best_policy \n"
	  }
   
### Take a sample of near policies
   my $temp_coef         = 1.0;
   my $counter           = 0;
   my $min_minus_mean    = $shared_mean_minus_min;
   my $policy_old        = $shared_policy_old;
   my $near_policies     = Math::MatrixReal->new($n_parameters,$n_parameters);
   my $near_policy_costs = Math::MatrixReal->new(1,$n_parameters);
   my $cur_policy_new = $initial_policy;
   my $cur_policy_new2 = $initial_policy;

while(1){
   for (my $sample=max($iter-$n_parameters,1);$sample<$iter;$sample++) {
        if ($temp_coef*$range > ($policy_history->row($sample) - $policy) * $covar_inv * ~($policy_history->row($sample) - $policy))
	  {$counter++;
	  $near_policies->assign_row($counter,$policy_history->row($sample));# print "$near_policies\n";
      $near_policy_costs->assign(1,$counter,$cost_history->element(1,$sample)); #print "$near_policy_costs\n";#print "$policy_history\n";
	  }
	  $temp_coef = $temp_coef * 3.0;
   }
      if ($counter > 1) {
      $near_bins_size = $counter;# print "Near bins:", "$near_bins_size\n";#print "Near policies:", "$near_policies \n";print "Near policy costs:", "$near_policy_costs\n";
      last;   }
	  else {$near_bins_size = 1; last;}
} #end while loop
     my $near_costs_block = Math::MatrixReal->new(1,$near_bins_size);
          for (my $i=1;$i<$near_bins_size+1;$i++) {
      $near_policy_costs->element(1,$i) > 0 ? $near_costs_block->assign(1,$i,$near_policy_costs->element(1,$i)) : $near_costs_block->assign(1,$i,1);
     }

###   Determine the minimal cost and the minimal index
   my $min_cost = min($near_costs_block->as_list());#  print "$near_costs_block\n"; print "$min_cost\n";
#  my $min_cost = $near_costs_block->minimum();#  print "$near_costs_block\n"; print "$min_cost\n";
   my $min_index;
          for (my $i=1;$i<$near_bins_size+1;$i++) {
      $min_index = $i unless $near_costs_block->element(1,$i) > $min_cost;
       }

###   Determine the minimal cost with ordinary vector (not MatrixReal one)
#      my @near_costs_block = ();
#          for (my $i=1;$i<$near_bins_size+1;$i++) {
#      $near_policy_costs->element(1,$i) > 0 ? push @near_costs_block, $near_policy_costs->element(1,$i) : push @near_costs_block,1;
#     }
#   my $min_cost = min(@near_costs_block);  print "@near_costs_block\n"; print "$min_cost\n";
#   my $min_index;
#          for (my $i=1;$i<$near_bins_size+1;$i++) {
#      $min_index = $i unless $near_costs_block->element(1,$i) > $min_cost;
#      } 
#   my $min_index;
#          for (my $i=0;$i<$near_bins_size;$i++) {
#      $min_index = $i+1 unless $near_costs_block[$i] > $min_cost;
#      }   
#      print "$min_index\n";

### Separate the near policies in a block and remove the zero vectors
   my $near_policy_block = Math::MatrixReal->new($near_bins_size,$n_parameters); 
          for (my $i=1;$i<$near_bins_size+1;$i++) {
       $near_policy_block->assign_row($i,$near_policies->row($i)); 
     } 
	# print "$near_policy_block\n";

### Determine the mean cost and subtract minimal cost from it
#  my $mean_cost = sum(@near_costs_block) / $near_bins_size; print "$mean_cost\n";
   my $mean_cost = sum($near_costs_block->as_list()) / $near_bins_size;# print "$mean_cost\n";
#  my $mean_cost = $near_costs_block->norm_sum() / $near_bins_size; print "$mean_cost\n";
   my $best_within_near_policies = $near_policy_block->row($min_index);# print "$best_within_near_policies\n";
   $shared_mean_minus_min = $mean_cost - $min_cost;	#print "$shared_mean_minus_min \n";
   my $mean_minus_min = $shared_mean_minus_min; #print "$mean_minus_min\n";

### Create a matrix for use in gradient descent instead of vector for loop
#   my $diffoftheta  = $near_policy_block - $policy_block; print "$diffoftheta\n";
#   my $diffscalar = ~$near_policy_block * $policy_block; print "$diffscalar\n";
#   my @sum = $diffscalar->as_list; print "$sum\n";

### Choose current policies and costs depending on the results from gradient descent
#    my $cur_policy_new = $initial_policy;
#    my $cur_policy_new2 = $initial_policy;
    if ($mean_minus_min != 0) {
    my ($e_cost,$cur_policy_new) = gradient_descent($near_costs_block,$near_policy_block,$mean_cost,$near_bins_size,$policy,$covar,$covar_inv);# print "$e_cost\n";
	#print "$e_cost\n";print "$cur_policy_new\n";

    my ($e_cost2,$cur_policy_new2) = gradient_descent($near_costs_block,$near_policy_block,$mean_cost,$near_bins_size,$best_within_near_policies,$covar,$covar_inv);# print "$e_cost\n";
		if ($e_cost > $e_cost2) {
			$cur_policy_new = $cur_policy_new2;
			$e_cost = $e_cost2;
		# print "$e_cost\n";#print "$cur_policy_new\n";
		}

	($e_cost2,$cur_policy_new2) = gradient_descent($near_costs_block,$near_policy_block,$mean_cost,$near_bins_size,$policy_eps,$covar,$covar_inv);# print "$e_cost\n";
		if ($e_cost > $e_cost2) {
			$cur_policy_new = $cur_policy_new2;
			$e_cost = $e_cost2;
		#print "$e_cost\n";print "$cur_policy_new\n";
		}

		if ($min_cost != $best_cost) {
    ($e_cost2,$cur_policy_new2) = gradient_descent($near_costs_block,$near_policy_block,$mean_cost,$near_bins_size,$best_policy,$covar,$covar_inv);# print "$e_cost\n";
		if ($e_cost > $e_cost2) {
			$cur_policy_new = $cur_policy_new2;
		# print "$e_cost\n";#print "$cur_policy_new\n";
		}
	}

# }
### Adjust sigma depending on the previous cost 
	$sigma = ($iter > 1 and $cost_history->element(1,$iter-1) > $cost_history->element(1,$iter)) ? ($sigma * (1.0 + $expansion_factor_sigma)) : ($sigma / ((1.0 + $expansion_factor_sigma) ** $imp_factor)); 
    print "$sigma\n"; # print "$expansion_factor_sigma\n";

			# STEP. Apply the result
        {
            lock($shared_lock);

            my $logLine = "$iter";
	     my $logLine1;
	     my $logLine2;

            foreach $row (@variables)
            {
                my $name = $row->[$VAR_NAME];

                $logLine  .= " ,$var_value{$name}";
#		 $logLine1 .= ",";
#        $logLine2 .= ", ";

           }
                $logLine2 = "$var_eng1";		

            print LOG "$logLine \n $logLine2 \n";
        }

### Adjust the covariant matrix
    if (($cur_policy_new-$policy) * $covar_inv * ~($cur_policy_new-$policy) < ($chiN * 1.5)**2) {
        $pc           = (1.0 - $cc) * $pc + $cc * ~($cur_policy_new-$policy) / $sigma;
        $c_normalized = (1.0 - $ccov) * $c_normalized + $pc * ~$pc * $ccov;
        $c_normalized = $c_normalized * 1.0 / $c_normalized->det() / $n_parameters if abs($c_normalized->det()) < 1.0e-20; #print $c_normalized->det();
    }

### Enforcing symmetry
#    for(my $k=1;$k<$n_parameters+1;$k++) {
#      for (my $i=$k+1;$i<$n_parameters+1;$i++) {
#        $c_normalized->element($k,$i) = $c_normalized->element($i,$k);
#	  }}
    $c_normalized = 0.5 * ($c_normalized + ~$c_normalized);

### Update the covariant matrix, its inverse and the current policy
    $covar        = $c_normalized * $sigma * $sigma;# print "$covar\n";
    $covar_inv    = $covar->inverse();#  print "$covar_inv\n";
    $policy       = $cur_policy_new; print "$cur_policy_new\n";

   $optimization_done = 1 if ($sigma < 1.0e-8);
 } 
 
 }
# print "Iteration: $iter\n, variable: $name, cost: $var_eng1{$name}, theta: $var_eng2{$name}, policy: $policy{$name}, policy_cur: $policy_cur{$name}\n";
}

    # STEP. Close log
    close(GAMELOG);
}

### SECTION. Simulating a game
sub simulate_ELO
{
    my ($var) = @_;
    my $ELO = 0.0;

    foreach my $key (keys(%$var))
    {
        my $a = -0.0001 * $variableIdx{$key}[$VAR_SIMUL_ELO];
        $ELO += $a * $var->{$key} ** 2;
    }
   
    return $ELO; 
}

sub simulate_winPerc
{
    my ($ELO_A, $ELO_B) = @_;

    my $Q_A = 10 ** ($ELO_A / 400);
    my $Q_B = 10 ** ($ELO_B / 400);

    return $Q_A / ($Q_A + $Q_B);
}

sub simulate_2games
{
    my ($var_eng1, $var_eng2) = @_;

    my $eng1_elo = simulate_ELO($var_eng1);
    my $eng2_elo = simulate_ELO($var_eng2);

    my $eng1_winperc = simulate_winPerc($eng1_elo, $eng2_elo);
    return $eng1_winperc * rand() * (rand() < $eng1_winperc ? 1 : -1);
#    return (rand() < $eng1_winperc ? 1 : -1) + (rand() < $eng1_winperc ? 1 : -1);
}

### SECTION. Playing a game
my @fenlines;
my ($eng1_pid, $eng2_pid);
local (*Eng1_Reader, *Eng1_Writer);
local (*Eng2_Reader, *Eng2_Writer);

sub engine_init
{
    # STEP. Read opening book.
    open(INPUT, "<$epd_path");
    binmode(INPUT);
    my @lines;
    (@lines) = <INPUT>;
    @fenlines = grep {/\w+/} @lines; # Filter out empty lines
    close (INPUT);
    die "epd read failure!" if ($#fenlines == -1);

    # STEP. Launch engines.
    $eng1_pid = open2(\*Eng1_Reader, \*Eng1_Writer, $eng1_path);
    $eng2_pid = open2(\*Eng2_Reader, \*Eng2_Writer, $eng2_path);

    # STEP. Init engines
    my $line;

    print Eng1_Writer "uci\n";
    print Eng2_Writer "uci\n";

    while(engine_readline(\*Eng1_Reader) ne "uciok") {} 
    while(engine_readline(\*Eng2_Reader) ne "uciok") {}
}

sub engine_quit 
{ 
    print Eng1_Writer "quit\n"; 
    print Eng2_Writer "quit\n"; 
    waitpid($eng1_pid, 0); 
    waitpid($eng2_pid, 0); 
} 

sub engine_readline
{
    my ($Reader) = @_;
    local $/ = $IS_WINDOWS ? "\r\n" : "\n";
    my $line = <$Reader>;
    chomp $line;
    return $line;
}

sub engine_2games
{ 
    my ($var_eng1,$var_eng2) = @_;
    my $result = 0;
	my $score = 0;
    my $line;
	my $cost;
	my $score_count = 0;
	my $sigmoid = 0;
#	my $var_eng1 = $shared_var_eng1;

    # STEP. Choose a random opening
    my $rand_i = int(rand($#fenlines + 1));
    my @tmparray = split(/\;/, $fenlines[$rand_i]);
    my $fenline = $tmparray[0];
    @tmparray = split(/ /, $fenline);
    my $side_to_start = $tmparray[1]; #'b' or 'w'

    # STEP. Send rounded values to engines
    my $column = 1;
    foreach my $var (keys(%$var_eng2))
    {   
		my $val1 = nearest(1, $var_eng1->element(1,$column));
        my $val2 = nearest(1, $var_eng2->{$var});
       
        print Eng1_Writer "setoption name $var value $val1\n";# print $val2;
        print Eng2_Writer "setoption name $var value $val2\n";
       $column++;
    }

    # STEP. Play two games
    for (my $eng1_is_white = 0; $eng1_is_white < 2; $eng1_is_white++)
    {
        # STEP. Tell engines to prepare for a new game
        print Eng1_Writer "ucinewgame\n";
        print Eng2_Writer "ucinewgame\n";

        print Eng1_Writer "isready\n";
        print Eng2_Writer "isready\n";

        # STEP. Wait for engines to be ready
        while(engine_readline(\*Eng1_Reader) ne "readyok") {}
        while(engine_readline(\*Eng2_Reader) ne "readyok") {}

        # STEP. Init Thinking times
        my $eng1_time = $base_time;
        my $eng2_time = $base_time;

        # STEP. Check which engine should start?
        my $engine_to_move = ($eng1_is_white == 1 && $side_to_start eq 'w') || ($eng1_is_white == 0 && $side_to_start eq 'b') ? 1 : 2;

        print GAMELOG "Starting game using opening fen: $fenline (opening line $rand_i). Engine to start: $engine_to_move\n";

        # STEP. Init game variabless
        my $moves = '';
        my $winner = 0;
        my $draw_shared_iter = 0;
        my @win_shared_iter  = (0, 0, 0);

GAME:  while(1)
       {
           my $wtime = nearest_floor(1, $eng1_is_white == 1 ? $eng1_time : $eng2_time);
           my $btime = nearest_floor(1, $eng1_is_white == 0 ? $eng1_time : $eng2_time);

           my $Curr_Writer = ($engine_to_move == 1 ? \*Eng1_Writer : \*Eng2_Writer);
           my $Curr_Reader = ($engine_to_move == 1 ? \*Eng1_Reader : \*Eng2_Reader);

           # STEP. Send engine the current positionn
           print $Curr_Writer "position fen $fenline" . ($moves ne '' ? " moves $moves" : "") . "\n";

           print GAMELOG "Engine " . ($engine_to_move == 1 ? '1' : '2') . " starts thinking. Time: " .
                  sprintf("%d", $engine_to_move == 1 ? $eng1_time : $eng2_time) . " Moves: $moves \n";

           # STEP. Let it go!
           my $t0 = time;
           print $Curr_Writer "go wtime $wtime btime $btime winc $inc_time binc $inc_time\n";

           # STEP. Read output from engine until it prints the bestmove.
           my $flag_mate = 0;
           my $flag_stalemate = 0;

READ:      while($line = engine_readline($Curr_Reader)) 
           {
               my @array = split(/ /, $line);

               # When engine is done, it prints bestmove.
               if ($#array >= 0 && $array[0] eq 'bestmove') {
                   
                   $flag_stalemate = 1 if ($array[1] eq '(none)');

                   $moves = $moves . " " . $array[1];
                   last READ;
               }

               # Check for mate in one
               if ($#array >= 9 && $array[0] eq 'info' && $array[1] eq 'depth' &&
                   $array[7] eq 'score' && $array[8] eq 'mate' && $array[9] eq '1') 
               {
                   $flag_mate = 1;
                   $winner = $engine_to_move;
               }

               # Record score
               if ($#array >= 7 && $array[0] eq 'info' && $array[1] eq 'depth' &&
                   $array[7] eq 'score') 
               { 
                   $score = $array[9] if ($array[8] eq 'cp');
#                   $score = +10000   if ($array[8] eq 'mate' && $array[9] > 0);
#                   $score = -10000   if ($array[8] eq 'mate' && $array[9] < 0);
                   $sigmoid = 1.0 / (1 + 10.0 ** -(0.7 * $score / 100))
                   
               }

		}

           print GAMELOG "Score: $score\n" if defined($score);

           # STEP. Update thinking times
           my $elapsed = time - $t0;
           $eng1_time = $eng1_time - ($engine_to_move == 1 ? $elapsed * 1000 - $inc_time : 0); 
           $eng2_time = $eng2_time - ($engine_to_move == 2 ? $elapsed * 1000 - $inc_time : 0);

           # STEP. Check for mate and stalemate
           if ($flag_mate)
           {
               $winner = $engine_to_move;
               last GAME;
           }

           if ($flag_stalemate)
           {
               $winner = 0;
               last GAME;
           }

           # STEP. Update draw shared_iter
           $draw_shared_iter = (abs($score) <= $draw_score_limit ? $draw_shared_iter + 1 : 0);

           print GAMELOG "Draw shared_iter: $draw_shared_iter / $draw_move_limit\n" if ($draw_shared_iter);

           if ($draw_shared_iter >= $draw_move_limit)
           {
               $winner = 0;
               last GAME;
           }

           # STEP. Update win shared_iters
           my $us   = $engine_to_move;
           my $them = $engine_to_move == 1 ? 2 : 1;

           $win_shared_iter[$us]   = ($score >= +$win_score_limit ? $win_shared_iter[$us]   + 1 : 0);
           $win_shared_iter[$them] = ($score <= -$win_score_limit ? $win_shared_iter[$them] + 1 : 0);
           
		   $score = $engine_to_move == 1 ? $score : 0; # print "$score\n";
		   $score_count++;
          
           print GAMELOG "Win shared_iter: $win_shared_iter[$us] / $win_move_limit\n" if ($win_shared_iter[$us]);
           print GAMELOG "Loss shared_iter: $win_shared_iter[$them] / $win_move_limit\n" if ($win_shared_iter[$them]); 
 
           if ($win_shared_iter[$us] >= $win_move_limit)
           {
               $winner = $us;
               last GAME;
           }

           if ($win_shared_iter[$them] >= $win_move_limit)
           {
               $winner = $them;
               last GAME;
           } 
#       $result = ($winner == 1 ? 1 : $winner == 2 ? 0 : 0.5); #print "$result\n"
#	   $score += $score;
#	   $cost = ($result - 1.0 / (1 + 10.0 ** -(0.7 * $score / 100))) ** 2; #print "$cost \n";

           # STEP. Change turn
           $engine_to_move = $them;
       }

       # STEP. Record the result
       print GAMELOG "Winner: $winner\n";
#	   print $score_count;
#      $result += ($winner == 1 ? 1 : $winner == 2 ? -1 : 0); #print "$result\n"
       $result = ($winner == 1 ? 1 : $winner == 2 ? 0 : 0.5); #print "$result\n"
	   $sigmoid += $sigmoid;
	   $cost = $score_count * ($result - $sigmoid) ** 2; #print "$cost \n";

   }

#	   $score += $score;  print "$score\n";
 	   $cost += $cost;   
       return print "$cost / $score_count \n";
}

######################################################
#sub take_sample 
#{ lock($shared_lock);
#   my ($policy,$policy_history,$cost_history) = @_;
#   my $temp_coef = 1.0;
#   my $counter = 0;
#   my $iter = $shared_iter;
#   my $min_minus_mean = $shared_mean_minus_min;
#   my $policy_old = $shared_policy_old;
#   my $near_policies     = Math::MatrixReal->new($n_parameters,$n_parameters);
#   my $near_policy_costs = Math::MatrixReal->new(1,$n_parameters);

#while(1){
#   for (my $sample=max($iter-$n_parameters,1);$sample<$iter;$sample++) {
#        if ($temp_coef*$range > ($policy_history->row($sample) - $policy) * $covar_inv * ~($policy_history->row($sample) - $policy))
#	  {$counter++;
#	  $near_policies->assign_row($counter,$policy_history->row($sample)); #print "$near_policies\n";
#      $near_policy_costs->assign(1,$counter,$cost_history->element(1,$sample)); #print "$near_policy_costs\n";#print "$policy_history\n";
#	  }
#	  $temp_coef = $temp_coef * 3.0;
#   }
#      if ($counter > 1) {
#      $near_bins_size = $counter; print "Near bins:", "$near_bins_size\n";#print "Near policies:", "$near_policies \n";print "Near policy costs:", "$near_policy_costs\n";
#      last;   }
#} #end while loop
#     my $near_costs_block = Math::MatrixReal->new(1,$near_bins_size);
#          for (my $i=1;$i<$near_bins_size+1;$i++) {
#      $near_policy_costs->element(1,$i) > 0 ? $near_costs_block->assign(1,$i,$near_policy_costs->element(1,$i)) : $near_costs_block->assign(1,$i,1);
#     }
#      my @near_costs_block = ();
#          for (my $i=1;$i<$near_bins_size+1;$i++) {
#      $near_policy_costs->element(1,$i) > 0 ? push @near_costs_block, $near_policy_costs->element(1,$i) : push @near_costs_block,1;
#     }
#   my $min_cost = min($near_costs_block->as_list());#  print "$near_costs_block\n"; print "$min_cost\n";
#  my $min_cost = $near_costs_block->minimum();#  print "$near_costs_block\n"; print "$min_cost\n";
#   my $min_index;
#          for (my $i=1;$i<$near_bins_size+1;$i++) {
#      $min_index = $i unless $near_costs_block->element(1,$i) > $min_cost;
#       }

#   my $min_cost = min(@near_costs_block);  print "@near_costs_block\n"; print "$min_cost\n";
#   my $min_index;
#          for (my $i=1;$i<$near_bins_size+1;$i++) {
#      $min_index = $i unless $near_costs_block->element(1,$i) > $min_cost;
#      } 
#   my $min_index;
#          for (my $i=0;$i<$near_bins_size;$i++) {
#      $min_index = $i+1 unless $near_costs_block[$i] > $min_cost;
#      }   
#      print "$min_index\n";

#   my $near_policy_block = Math::MatrixReal->new($near_bins_size,$n_parameters); 
#          for (my $i=1;$i<$near_bins_size+1;$i++) {
#       $near_policy_block->assign_row($i,$near_policies->row($i)); 
#     } 
	# print "$near_policy_block\n";
    
#   my $mean_cost = sum(@near_costs_block) / $near_bins_size; print "$mean_cost\n";
#   my $mean_cost = sum($near_costs_block->as_list()) / $near_bins_size; #print "$mean_cost\n";
#  my $mean_cost = $near_costs_block->norm_sum() / $near_bins_size; print "$mean_cost\n";
#   my $best_within_near_policies = $near_policy_block->row($min_index);# print "$best_within_near_policies\n";
#   $shared_mean_minus_min = $mean_cost - $min_cost;	#print "$shared_mean_minus_min \n";
#   my $mean_minus_min = $shared_mean_minus_min; #print "$mean_minus_min\n";
#   my $policy_block = Math::MatrixReal->new($near_bins_size,$n_parameters); 
#          for (my $i=1;$i<$near_bins_size+1;$i++) {
#       $policy_block->assign_row($i,$near_policies->row($i)); 
#		  }
#      print "$policy_block\n";
#   my $diffoftheta  = $near_policy_block - $policy_block; print "$diffoftheta\n";
#   my $diffscalar = ~$near_policy_block * $policy_block; print "$diffscalar\n";
#   my @sum = $diffscalar->as_list; print "$sum\n";
#	my ($e_cost,$cur_policy_new) = gradient_descent($near_costs_block,$near_policy_block,$mean_cost,$near_bins_size,$policy_old,$policy,$covar,$covar_inv);# print "$e_cost\n";
		#print "$cur_policy_new\n";

#   return ($near_bins_size,$near_policy_costs,$near_policies);   
#} #end while
#} # end take_sample
#######################################################################

    ### Natural gradient descent

sub gradient_descent {

lock($shared_lock);
my $iter = $shared_iter;
my $mean_minus_min = $shared_mean_minus_min;
my ($near_costs_block,$near_policy_block,$mean_cost,$near_bins_size,$policy,$covar,$covar_inv) = @_; 
    my $op_cost;

#if ($mean_minus_min != 0) {

  my ($Pprior,$a,$b,$expMD2,$probabilistic_distance,$op_policy,$residual_term,$adot,$bdot,$Jaco,$update,$diffoftheta);
  my $policy_old = $policy;

### $Pprior        = 2.0 / pow(sqrt(2.0 * M_PI), n_parameter);
    $Pprior        = 1.0;

    my $terminate  = 0;

    ### STEP: Autotune alpha according to 50% in every step criteria
#    my $covar_factored     = $covar * $cost2policy_cov_factor; 
    my $alpha              = 10.0 * $covar * $cost2policy_cov_factor / $mean_minus_min;
#    my $covar_inv          = $covar->inverse();

    for(my $i = 0; $i < 200; $i++){

       $a = $Pprior * $mean_cost;
       $b = $Pprior;

       for(my $row=1;$row<$near_bins_size+1;$row++){
       
       $diffoftheta       = $near_policy_block->row($row) - $policy_old; #print "$diffoftheta\n";
       $residual_term     = 0.5 * $diffoftheta * $covar_inv / $cost2policy_cov_factor; #print "$residual_term\n";
    my $diff_covar        = ~(-0.25 * $diffoftheta * $covar_inv / $cost2policy_cov_factor); #print "$diff_covar\n";
    my $MD2               = $diff_covar->scalar_product(~$diffoftheta);# print "$MD2\n";
       $expMD2            = exp $MD2; #print "$expMD2\n";
       $adot              = $near_costs_block->element(1,$row) * $expMD2 * $residual_term; #print "$adot\n";
       $bdot              = $expMD2 * $residual_term; #print "$bdot\n";
       $a                += $near_costs_block->element(1,$row) * $expMD2; #print "$a\n";
       $b                += $expMD2; #print "$b\n";
   
      last if ($terminate);
#      $covar_inv              = $covar->inverse();
       $Jaco                   = $adot / $b - $bdot * $a / $b / $b; #print "$Jaco\n"; print "$alpha\n";
       $update                 = $Jaco * $alpha; #print "$update\n";
       $op_policy              = $policy_old - $update; #print "$op_policy\n";
       $probabilistic_distance = $update * $covar_inv / $cost2policy_cov_factor * ~$update;# print "$probabilistic_distance\n";
       $policy_old             = $op_policy;
 
       $terminate = 1 if ($probabilistic_distance < 0.0001 || ($op_policy-$policy) * $covar_inv / $cost2policy_cov_factor * ~($op_policy-$policy) > 1.0);
    } #end for inner clause
    $op_cost = $a / $b; 
  } #end for outer clause
return ($op_cost,$policy_old); 
#} # end if clause
} # end gradient_descent
