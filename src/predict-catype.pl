#!/usr/bin/perl

use strict;
use warnings;


my $prob_submodel_crit = 0.25; 
my $prob_crit = 0.05;

use Cwd;

my $cwd = getcwd();

use lib '.';
use Common;
use threads;
use threads::shared;
use POTTRConfig;
use CancerTypes;
CancerTypes::ON_DEMAND_INIT();
chdir $cwd;

sub get_sane_catype_path(@) {
	my @catypes = grep { defined } @_;
	s/[[:punct:] ]+/_/g for @catypes;
	return join("--", @catypes);
}

sub get_sane_classifier_path_name { return $_[0] =~ s/[[:space:]\-]//gr  }


my $model_type = '' ; # shift @ARGV;
my $mlev_cmd = './mlev.pl';
# $mlev_cmd = $ENV{'HOME'}."/src/mlfw/mlev.pl" if ! -x $mlev_cmd;
die "Unable to locate mlev.pl" if ! -x $mlev_cmd;


my $nproc = `nproc`;
chomp $nproc;

sub run {
# 	print STDERR "\e[1;37m> $_[0]\e[0m\n";
	return qx{$_[0]};
}

sub do_predict($$) {
	my $model_file = shift;
	my $data_file = shift;
	my @output = run "$mlev_cmd predict $model_file $data_file 2> /dev/null"; #  
# 	print STDERR map { my $a = $_; $a =~ s/\n//; "\e[1;32m$a\e[0m\n" } ($model_file, @output) ;
# 	print STDERR $model_file;
	die if ! -f $model_file;
	chomp for @output;
	my ($header, $data) = @output;
# 	print STDERR @output;
	my %a;
	@a{ (split /\t/, $header) } = (split /\t/, $data) ;
	warn "$model_file contains an empty header.\n" if ! defined $header;
	return join("\t", $model_file , $a{'Y'} // 'NA')."\n";
}


my @profile = split / +/, join(' ', @ARGV);

my $tmpfile = '/tmp/camuts.tsv';
my @outlines ;
push @outlines, join("\t", @profile, '__class__')."\n";
push @outlines, join("\t", ( map {1} (0..scalar(@profile)) ) )."\n";

open OUTF, ">$tmpfile";
print OUTF @outlines;
close OUTF;



my @retval; 
my @retval_sm; 
my $cpucount = 0;

if ( $model_type =~ /,/ ) {
	$model_type =~ s/,/|/g;
	$model_type = "(?:$model_type)";
}

# my @models = grep { /SVC.*D1|LogisticRegression-/} grep { !/unknown|-carcinoma.model/i } dirr('camuts') ;
my @models = grep { /LogisticRegression-/} grep { !/unknown|-carcinoma.model/i } dirr('camuts') ;
# my @models = grep { /SVC.*D1.*-/} grep { !/unknown|-carcinoma.model/i } dirr('camuts') ;

my @valid_models = sort grep { /$model_type.*model/ } grep { ! /--/ }  @models ;
my %valid_catypes_fn;
for my $m (@valid_models)  {
	$m =~ /\/[A-Za-z0-9\.]+-(.*).model/; 
	$valid_catypes_fn{$1}{$m} = 1;
}


my @valid_submodels = sort grep { /$model_type.*model/ } grep { /--/ } @models ;
my %valid_catypes_fn_sm;
for my $m (@valid_submodels)  {
	$m =~ /\/[A-Za-z0-9\.]+-(.*).model/; 
	my $sm = $1;
	$sm =~ s/--/\|/g;	
	$valid_catypes_fn_sm{$sm}{$m} = 1;
}


my $n_completed =0;
my $n_total = scalar(@valid_models);
my $start_time = time();

sub print_progress {
	my $notes = shift // '';
	$notes =~ s/camuts\/(?:(?:SVC|LogisticRegression|RandomForest)[\w\d\.]*)-|\.model|[\n\r\t].*//g ;
	$notes =~ s/_/ /g ; 
	my $f_completed = $n_completed / $n_total ;
	my $perc_completed = sprintf("%3.0f%%", $f_completed * 100);
	my $length = 40;
	my $length_completed = int( $f_completed * $length + 0.5) ;
	my $elapsed = time() - $start_time ;
	my $ETA_s = 'NA';
	if ( $f_completed > 0 ) {
		$ETA_s = sprintf "%.0f s", ( $elapsed / $f_completed ) * (1 - $f_completed);
	}
	
	print STDERR "Completed: $n_completed / $n_total ($perc_completed) [". join('', map { ($_ < $length_completed ? "=" : "." )} (0..($length-1)))."] ETA: $ETA_s  $notes\e[K\r";
}



my @wanted_submodels; 

sub post_process_results {
	my $arg = shift;
	push @retval, $arg ;
	++$n_completed;
	my ($t, $p) = split /[\t\n]/, $arg;
	if ( $p >= $prob_submodel_crit) {
		push @wanted_submodels, $t;
	}
	print_progress( $arg );
	return $arg;
}

sub post_process_results_submodel {
	my $arg = shift;
	push @retval_sm, $arg ;
	++$n_completed;
	my ($t, $p) = split /[\t\n]/, $arg;
	print_progress( $arg );
	return $arg;
}


# for my $catype (sort keys %valid_catypes_fn) {
# 	my $best_sclf = $best_clf{get_sane_catype_path($catype)} ;
# 	my ($model_file) = grep { /\b$best_sclf-/ } @valid_models ;
	
	for my $model_file ( @valid_models ) {
# 		$valid_catypes_fn{$model_file} = $catype;
	# 	do_predict($model_file, $tmpfile);
		$cpucount ++;
		my ($thr) = threads->create( \&do_predict, $model_file, $tmpfile);
#		if ( $cpucount >= 4 * $nproc ) {
#			for my $thr ( threads->list() ) {
#				--$cpucount;
#				post_process_results( $thr -> join() );
#			}
#		}
	}
# }

post_process_results( $_ -> join() ) for threads->list() ;

for my $sm (@wanted_submodels) {
	$sm =~ s/camuts\/(?:(?:SVC|LogisticRegression|LogisticRegressionPelasticnet|RandomForest)[\w\d\.]*)-|\.model|[\n\r\t].*//g ;
	for my $vsm ( grep { /\b$sm--/ } @valid_submodels ) {
		my ($thr) = threads->create( \&do_predict, $vsm, $tmpfile);
#		if ( $cpucount >= 4 * $nproc ) {
#			for my $thr ( threads->list() ) {
#				--$cpucount;
#				post_process_results_submodel( $thr -> join() );
#			}
#		}
	}
}

post_process_results_submodel( $_ -> join() ) for threads->list() ;

print STDERR "\n";

sub process_retval(\%\%\@) {
	my $prob = $_[0];
	my $prob_clf = $_[1];
	my @retval = @{ $_[2] };

	for my $line (@retval) {
		chomp $line ;
		$line =~ s/camuts\/(SVC|LogisticRegression|RandomForest)[\w\d\.]*-//; 
		my $model_name = $1;
	# 	print STDERR "$model_name\t$model_type\n"; 
		$line =~ s/\.model//;
		$line =~ s/_/ /g; 
		my ($t, $p) = split /\t/, $line; 
		my $tt = join( '|', map { CancerTypes::get_preferred_catype_term( $_ ) } (split /--/, $t) );
		if ( ! exists $$prob{$tt} or $$prob{$tt} < $p ) {
			$$prob{$tt} = $p ;
			($$prob_clf{$tt}) = ( $model_name =~ /($model_type)/ );
		}
	}
}

sub hl_prob($$) {
	my $text = shift;
	my $prob = shift;
	my $col = int( min( ($prob / 0.667), 1 ) * 12 ) + 243;
	return "\e[38;5;${col}m$text\e[0m" ;
	return $text;
}

my (%prob, %prob_clf);
my (%prob_sm, %prob_clf_sm);

process_retval(%prob, %prob_clf, @retval);
process_retval(%prob_sm, %prob_clf_sm, @retval_sm);

for my $catype (sort { $prob{$b} <=> $prob{$a} } grep { $prob{$_} >= $prob_crit } keys %prob) {
	print join("\t", sprintf("%.6f", $prob{$catype}), hl_prob( $catype, $prob{$catype}) )."\n"  ; # $prob_clf{$catype} // '', 
	if ( $prob{$catype} > $prob_submodel_crit ) {
		my @vsm = grep { /^$catype\|/ } keys %prob_sm;
		for my $vsm ( sort { $prob_sm{$b} <=> $prob_sm{$a} } @vsm) {
			next if $prob_sm{$vsm} < $prob_crit ;
			print join("\t", sprintf("|- %.3f", $prob_sm{$vsm}), hl_prob( "  |- ".$vsm, $prob_sm{$vsm}) )."\n"  ; # $prob_clf_sm{$vsm} // '', 
		}
	}
}

exit;

